/**
 * emg-record-plotter firmware -- raw 4-channel EMG exfiltration (ESP32-S3)
 *
 * Continuous-mode ADC (DMA) at ADC_SAMPLE_FREQ_HZ total (default 80 kHz ->
 * 20 kHz per channel), raw 12-bit codes, no processing on the device.
 * Streamed over UDP in the same wire format as emg-hci's exfiltrator /
 * ML firmwares, so viewer/emg_plotter.py and emg_classifier_ui can read it:
 *
 *   [device_id:1][seq:4 BE][timestamp_us:4 BE][num_channels:1]
 *   [SAMPLES_PER_PACKET x NUM_CH x int32 BE]
 *
 * Tasks
 *   adc  (core 1, prio 10)  drains the ADC DMA pool, assembles packets and
 *                           queues them.  Never touches a socket.
 *   send (core 0, prio 5)   sendto(); on ENOMEM (WiFi TX buffers full) it
 *                           yields a tick and retries a few times.
 *   main                    one status line per second.
 *
 * The sequence number is assigned when a packet is assembled, so every loss
 * (queue drop, failed send, air) shows up as a seq gap on the PC.
 *
 * Ported from emg-hci/firmware/emg_exfiltrator_node/src/esp32_mini_4ch.cpp.
 */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "esp_adc/adc_continuous.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"   // xQueueCreateWithCaps
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
#include "soc/soc_caps.h"

static const char *TAG = "EMG";

// ============================================================================
// CONFIG (each value can be overridden from platformio.ini with -D...)
// ============================================================================
#ifndef DEVICE_ID
#define DEVICE_ID 6
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "GL-NET"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS "goodlife"
#endif
#ifndef UDP_TARGET_IP
#define UDP_TARGET_IP "192.168.7.240"     // PC running the viewer / GUI
#endif
#ifndef UDP_TARGET_PORT
#define UDP_TARGET_PORT 5005
#endif
// Total ADC conversions per second over all channels.  ESP32-S3 maximum is
// SOC_ADC_SAMPLE_FREQ_THRES_HIGH = 83333.  80000 (20 kHz/ch) matches every
// earlier recording and the trained models.
#ifndef ADC_SAMPLE_FREQ_HZ
#define ADC_SAMPLE_FREQ_HZ 80000
#endif
// Wire format
//   2 = compact (default): samples as uint16 BE, many samples per packet.
//       num_channels byte = 0x80 | n_channels marks it.  viewer only.
//   1 = legacy: samples as int32 BE, <= 31 samples/packet (<= 512 B), readable
//       by emg_classifier_ui and the emg-hci visualizers.
// Measured on the N16R8 (RSSI -22 dBm, ch 11): legacy at 807 frames/s only got
// ~550 frames/s out, the rest failed with ENOMEM; compact needs ~126 frames/s.
#ifndef WIRE_FORMAT
#define WIRE_FORMAT 2
#endif
#ifndef SAMPLES_PER_PACKET
#if WIRE_FORMAT == 2
#define SAMPLES_PER_PACKET 160     // 8 ms at 20 kHz, 1290 B, ~126 pkt/s, ~1.3 Mbit/s
#else
#define SAMPLES_PER_PACKET 25      // ~800 pkt/s, 410 B, ~2.6 Mbit/s
#endif
#endif
// WiFi TX power in dBm (2..20); 0 = driver default (max ~20 dBm).  Lower it
// when the board sits right next to the router: a too-strong signal can
// overload the router's receiver and force retransmissions.
#ifndef WIFI_TX_POWER_DBM
#define WIFI_TX_POWER_DBM 0
#endif
// ADC1 channel numbers in logical CH0..CHn order.
#ifndef EMG_ADC_CHANNELS
#define EMG_ADC_CHANNELS 0, 1, 2, 3        // DevKitC: GPIO1..GPIO4
#endif

static constexpr int ADC_CH_NUM[] = {EMG_ADC_CHANNELS};
static constexpr int NUM_CH = sizeof(ADC_CH_NUM) / sizeof(ADC_CH_NUM[0]);
static constexpr int PER_CH_RATE = ADC_SAMPLE_FREQ_HZ / NUM_CH;

#if WIRE_FORMAT == 2
typedef uint16_t sample_t;
static constexpr int SAMPLE_BYTES = 2;
#else
typedef int32_t sample_t;
static constexpr int SAMPLE_BYTES = 4;
#endif

static_assert(ADC_SAMPLE_FREQ_HZ <= SOC_ADC_SAMPLE_FREQ_THRES_HIGH, "ADC rate above hardware max");
static_assert(SAMPLES_PER_PACKET * NUM_CH * SAMPLE_BYTES + 10 <= 1460, "packet larger than one WiFi frame");

// ADC DMA: one conversion frame = 64 results per channel; pool = 16 frames
// (~51 ms at 80 kHz) so short WiFi / scheduling hiccups never overflow it.
static constexpr uint32_t CONV_FRAME_BYTES = NUM_CH * 64 * SOC_ADC_DIGI_RESULT_BYTES;
static constexpr uint32_t POOL_BYTES       = CONV_FRAME_BYTES * 16;
static constexpr int      WIRE_BYTES       = 10 + SAMPLES_PER_PACKET * NUM_CH * SAMPLE_BYTES;
// Stall buffer: WiFi on this setup stops for ~1 s every 20-40 s (measured).
// 2 s of packets in PSRAM let the sender catch up afterwards instead of
// dropping data; falls back to a small internal-RAM queue without PSRAM.
#ifndef STALL_BUFFER_MS
#define STALL_BUFFER_MS 2000
#endif
static constexpr int QUEUE_DEPTH_PSRAM = (PER_CH_RATE / SAMPLES_PER_PACKET) * STALL_BUFFER_MS / 1000;
static constexpr int QUEUE_DEPTH_IRAM  = (WIRE_FORMAT == 2) ? 32 : 64;

// ============================================================================
// STATE
// ============================================================================
typedef struct {
    uint32_t seq;
    uint32_t ts_us;                               // esp_timer at first sample
    sample_t s[SAMPLES_PER_PACKET][NUM_CH];
} pkt_t;

static adc_continuous_handle_t s_adc = NULL;
static TaskHandle_t            s_adc_task = NULL;
static QueueHandle_t           s_queue = NULL;
static int                     s_queue_depth = 0;
static int                     s_sock = -1;
static struct sockaddr_in      s_dest;
static SemaphoreHandle_t       s_wifi_sem = NULL;

// counters (single writer each; read once per second for the status line)
static uint32_t s_samples = 0, s_seq = 0, s_qdrop = 0;
static uint32_t s_tx_ok = 0, s_tx_fail = 0, s_tx_retry = 0;
static int      s_tx_errno = 0;
static volatile uint32_t s_pool_ovf = 0;          // written from ISR

// per-channel level stats, published by the ADC task about once per second
typedef struct { uint32_t n; int32_t min[NUM_CH], max[NUM_CH], mean[NUM_CH]; float clip[NUM_CH]; } levels_t;
static levels_t    s_levels;
static portMUX_TYPE s_levels_mux = portMUX_INITIALIZER_UNLOCKED;

// ============================================================================
// WIFI
// ============================================================================
static void wifi_event(void *, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected -- reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "WiFi connected, IP " IPSTR, IP2STR(&ev->ip_info.ip));
        xSemaphoreGive(s_wifi_sem);
    }
}

static void wifi_init(void)
{
    s_wifi_sem = xSemaphoreCreateBinary();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL, NULL));

    wifi_config_t wc = {};
    strncpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password) - 1);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Connecting to \"%s\" ...", WIFI_SSID);

    if (xSemaphoreTake(s_wifi_sem, pdMS_TO_TICKS(15000)) != pdTRUE) {
        ESP_LOGW(TAG, "WiFi not connected yet -- streaming starts once it is");
    }
    esp_wifi_set_ps(WIFI_PS_NONE);                 // no modem sleep while streaming
    if (WIFI_TX_POWER_DBM > 0) {
        esp_wifi_set_max_tx_power((int8_t)(WIFI_TX_POWER_DBM * 4));   // 0.25 dBm units
    }
    int8_t txp = 0;
    esp_wifi_get_max_tx_power(&txp);
    wifi_ap_record_t ap;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
        ESP_LOGI(TAG, "AP RSSI %d dBm, channel %d, TX power %.1f dBm", ap.rssi, ap.primary, txp / 4.0f);
    }
}

static int wifi_rssi(void)
{
    wifi_ap_record_t ap;
    return (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) ? ap.rssi : 0;
}

// ============================================================================
// ADC
// ============================================================================
static bool IRAM_ATTR on_conv_done(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *)
{
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(s_adc_task, &woken);
    return woken == pdTRUE;
}

static bool IRAM_ATTR on_pool_ovf(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *)
{
    s_pool_ovf = s_pool_ovf + 1;                  // DMA data lost (ADC task too slow)
    return false;
}

static void adc_init(void)
{
    adc_continuous_handle_cfg_t hcfg = {};
    hcfg.max_store_buf_size = POOL_BYTES;
    hcfg.conv_frame_size    = CONV_FRAME_BYTES;
    ESP_ERROR_CHECK(adc_continuous_new_handle(&hcfg, &s_adc));

    adc_digi_pattern_config_t pattern[NUM_CH];
    for (int i = 0; i < NUM_CH; i++) {
        pattern[i].atten     = ADC_ATTEN_DB_12;    // ~0..3.1 V
        pattern[i].channel   = (uint8_t)ADC_CH_NUM[i];
        pattern[i].unit      = ADC_UNIT_1;
        pattern[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    }
    adc_continuous_config_t dcfg = {};
    dcfg.pattern_num    = NUM_CH;
    dcfg.adc_pattern    = pattern;
    dcfg.sample_freq_hz = ADC_SAMPLE_FREQ_HZ;
    dcfg.conv_mode      = ADC_CONV_SINGLE_UNIT_1;
    dcfg.format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2;
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &dcfg));

    adc_continuous_evt_cbs_t cbs = {};
    cbs.on_conv_done = on_conv_done;
    cbs.on_pool_ovf  = on_pool_ovf;
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(s_adc, &cbs, NULL));

    ESP_LOGI(TAG, "ADC: %d ch, %d Hz total (%d Hz/ch), 12-bit, 12 dB atten",
             NUM_CH, ADC_SAMPLE_FREQ_HZ, PER_CH_RATE);
}

static inline int ch_index(uint32_t hw)
{
    for (int i = 0; i < NUM_CH; i++) if (ADC_CH_NUM[i] == (int)hw) return i;
    return -1;
}

static void adc_task(void *)
{
    static uint8_t buf[CONV_FRAME_BYTES];
    static pkt_t   pkt;
    int32_t cur[NUM_CH] = {};
    bool    filled[NUM_CH] = {};
    int     n = 0;

    // level-stat accumulators
    int32_t lmin[NUM_CH], lmax[NUM_CH];
    int64_t lsum[NUM_CH] = {};
    uint32_t lclip[NUM_CH] = {}, ln = 0;

    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
    ESP_LOGI(TAG, "Sampling started");

    while (true) {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
        uint32_t got = 0;
        while (adc_continuous_read(s_adc, buf, sizeof(buf), &got, 0) == ESP_OK) {
            for (uint32_t i = 0; i < got; i += SOC_ADC_DIGI_RESULT_BYTES) {
                const adc_digi_output_data_t *p = (const adc_digi_output_data_t *)&buf[i];
                int idx = ch_index(p->type2.channel);
                if (idx < 0) continue;
                cur[idx] = (int32_t)p->type2.data;
                filled[idx] = true;
                bool all = true;
                for (int c = 0; c < NUM_CH; c++) all = all && filled[c];
                if (!all) continue;

                // one complete multi-channel sample
                if (n == 0) pkt.ts_us = (uint32_t)esp_timer_get_time();
                for (int c = 0; c < NUM_CH; c++) {
                    int32_t v = cur[c];
                    pkt.s[n][c] = (sample_t)v;
                    filled[c] = false;
                    if (ln == 0 || v < lmin[c]) lmin[c] = v;
                    if (ln == 0 || v > lmax[c]) lmax[c] = v;
                    lsum[c] += v;
                    if (v <= 0 || v >= 4095) lclip[c]++;
                }
                ln++;
                s_samples++;

                if (++n == SAMPLES_PER_PACKET) {
                    pkt.seq = s_seq++;
                    if (xQueueSend(s_queue, &pkt, 0) != pdTRUE) {   // sender behind:
                        static pkt_t old;                           // drop oldest
                        xQueueReceive(s_queue, &old, 0);
                        s_qdrop++;
                        xQueueSend(s_queue, &pkt, 0);
                    }
                    n = 0;
                }

                if (ln >= (uint32_t)PER_CH_RATE) {                 // ~1 s: publish
                    taskENTER_CRITICAL(&s_levels_mux);
                    s_levels.n = ln;
                    for (int c = 0; c < NUM_CH; c++) {
                        s_levels.min[c]  = lmin[c];
                        s_levels.max[c]  = lmax[c];
                        s_levels.mean[c] = (int32_t)(lsum[c] / ln);
                        s_levels.clip[c] = 100.0f * lclip[c] / ln;
                        lsum[c] = 0; lclip[c] = 0;
                    }
                    taskEXIT_CRITICAL(&s_levels_mux);
                    ln = 0;
                }
            }
        }
    }
}

// ============================================================================
// UDP SENDER
// ============================================================================
static inline void put_be32(uint8_t *b, uint32_t v)
{
    b[0] = v >> 24; b[1] = v >> 16; b[2] = v >> 8; b[3] = v;
}

static void send_task(void *)
{
    static pkt_t   pkt;
    static uint8_t wire[WIRE_BYTES];

    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    memset(&s_dest, 0, sizeof(s_dest));
    s_dest.sin_family = AF_INET;
    s_dest.sin_port   = htons(UDP_TARGET_PORT);
    inet_pton(AF_INET, UDP_TARGET_IP, &s_dest.sin_addr);
    ESP_LOGI(TAG, "UDP -> %s:%d  device_id=%d  %s format, %d samples/pkt (%d B, %d pkt/s)",
             UDP_TARGET_IP, UDP_TARGET_PORT, DEVICE_ID, WIRE_FORMAT == 2 ? "compact u16" : "legacy i32",
             SAMPLES_PER_PACKET, WIRE_BYTES, PER_CH_RATE / SAMPLES_PER_PACKET);

    while (true) {
        xQueueReceive(s_queue, &pkt, portMAX_DELAY);
        size_t off = 0;
        wire[off++] = DEVICE_ID;
        put_be32(&wire[off], pkt.seq);   off += 4;
        put_be32(&wire[off], pkt.ts_us); off += 4;
#if WIRE_FORMAT == 2
        wire[off++] = 0x80 | NUM_CH;                  // compact-format marker
        for (int s = 0; s < SAMPLES_PER_PACKET; s++)
            for (int c = 0; c < NUM_CH; c++) {
                wire[off++] = pkt.s[s][c] >> 8;
                wire[off++] = pkt.s[s][c] & 0xFF;
            }
#else
        wire[off++] = NUM_CH;
        for (int s = 0; s < SAMPLES_PER_PACKET; s++)
            for (int c = 0; c < NUM_CH; c++) { put_be32(&wire[off], (uint32_t)pkt.s[s][c]); off += 4; }
#endif

        // ENOMEM = WiFi TX buffers full (link busy/stalled): keep this packet
        // and retry; the stall buffer absorbs the backlog.  Only other errors
        // drop the packet.
        while (true) {
            if (sendto(s_sock, wire, off, 0, (struct sockaddr *)&s_dest, sizeof(s_dest)) >= 0) {
                s_tx_ok++;
                break;
            }
            if (errno != ENOMEM) {
                s_tx_fail++;
                s_tx_errno = errno;
                break;
            }
            s_tx_retry++;
            vTaskDelay(1);                             // let the WiFi task drain
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "emg-record-plotter firmware  device %d  %d ch", DEVICE_ID, NUM_CH);

    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        r = nvs_flash_init();
    }
    ESP_ERROR_CHECK(r);

    wifi_init();
    s_queue = xQueueCreateWithCaps(QUEUE_DEPTH_PSRAM, sizeof(pkt_t), MALLOC_CAP_SPIRAM);
    s_queue_depth = QUEUE_DEPTH_PSRAM;
    if (s_queue == NULL) {
        s_queue = xQueueCreate(QUEUE_DEPTH_IRAM, sizeof(pkt_t));
        s_queue_depth = QUEUE_DEPTH_IRAM;
    }
    ESP_LOGI(TAG, "Stall buffer: %d packets (%d ms) in %s", s_queue_depth,
             s_queue_depth * SAMPLES_PER_PACKET * 1000 / PER_CH_RATE,
             s_queue_depth == QUEUE_DEPTH_PSRAM ? "PSRAM" : "internal RAM (no PSRAM)");
    adc_init();

    xTaskCreatePinnedToCore(send_task, "send", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(adc_task, "adc", 4096, NULL, 10, &s_adc_task, 1);

    // Status line once per second
    uint32_t p_samples = 0, p_ok = 0, p_fail = 0, p_retry = 0, p_qdrop = 0, p_ovf = 0;
    int64_t  p_t = esp_timer_get_time();
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        int64_t now = esp_timer_get_time();
        float dt = (now - p_t) / 1e6f;
        p_t = now;
        uint32_t sm = s_samples, ok = s_tx_ok, fl = s_tx_fail, rt = s_tx_retry, qd = s_qdrop, ov = s_pool_ovf;
        char err[40] = "";
        if (fl != p_fail) snprintf(err, sizeof(err), " (%s)", strerror(s_tx_errno));
        ESP_LOGI(TAG, "%.0f S/s/ch | tx %.0f pkt/s, retry %lu, fail %lu%s, qdrop %lu, buf %u/%d, dma-ovf %lu | RSSI %d",
                 (sm - p_samples) / dt, (ok - p_ok) / dt, (unsigned long)(rt - p_retry),
                 (unsigned long)(fl - p_fail), err, (unsigned long)(qd - p_qdrop),
                 (unsigned)uxQueueMessagesWaiting(s_queue), s_queue_depth,
                 (unsigned long)(ov - p_ovf), wifi_rssi());
        p_samples = sm; p_ok = ok; p_fail = fl; p_retry = rt; p_qdrop = qd; p_ovf = ov;

        levels_t lv;
        taskENTER_CRITICAL(&s_levels_mux);
        lv = s_levels;
        taskEXIT_CRITICAL(&s_levels_mux);
        if (lv.n) {
            char line[200];
            int o = 0;
            for (int c = 0; c < NUM_CH && o < (int)sizeof(line); c++)
                o += snprintf(line + o, sizeof(line) - o, "%sCH%d %ld/%ld/%ld %.0f%%", c ? " | " : "",
                              c, (long)lv.min[c], (long)lv.mean[c], (long)lv.max[c], lv.clip[c]);
            ESP_LOGI(TAG, "min/mean/max clip: %s", line);
        }
    }
}
