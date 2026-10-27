# BioSphere AI

**Telefonla çekilen videodan 360° sanal tur — açık kaynak.**

[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-2dd4bf.svg)](LICENSE)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-3776ab.svg)
![Next.js 16](https://img.shields.io/badge/next.js-16-000.svg)
![Durum: kişisel proje](https://img.shields.io/badge/durum-ki%C5%9Fisel%20proje-f59e0b.svg)

[Proje sitesi](https://biosphere.bilalgurkansanli.com) · [English README](README.md) · [Mimari notları](docs/ARCHITECTURE.md) · [Katkı rehberi](CONTRIBUTING.md)

<table>
  <tr>
    <td><img src="web/public/shots/player.webp" alt="360 oynatıcı: odalar, kapı eşiğinde zemin halkası, araç çubuğu"></td>
    <td><img src="web/public/shots/plan.webp" alt="Panoramalardan çıkarılan kuş bakışı kat planı"></td>
  </tr>
  <tr>
    <td><img src="web/public/shots/capture.webp" alt="Oda ekleme: telefonun giriş yapmadan çekim yaptığı QR kod"></td>
    <td><img src="web/public/shots/vr.webp" alt="Telefonla VR gözlük modu: mercek düzeltmeli stereo görüntü ve bakarak seçme"></td>
  </tr>
</table>

<sub>Bütün ekran görüntüleri motorun sentetik test odalarından; bu depoda hiçbir gerçek ev yok.</sub>

Emlakçı odanın ortasında durur, telefonu bir dakika boyunca kendi etrafında çevirir. BioSphere
videodan en keskin kareleri seçer, kameranın nasıl döndüğünü çözer, 8K'ya kadar bir
equirectangular panorama diker ve bunu odaları birbirine bağlayan, kat planı çizen, mesafe
ölçen, VR'da açılan ve herhangi bir ilan sayfasına `<iframe>` ile gömülebilen bir web
oynatıcısında sunar. 360 kamera ya da özel donanım gerekmez.

Emlakçılara satılacak bir ürün olarak başladı; şimdi tek geliştiricili, açık bir mühendislik
projesi. Buradaki her şey tek bir dizüstü bilgisayarda çalışıyor.

---

## Neler hazır, neler gelişiyor

Ekim 2026 itibarıyla durum. Sayılar tahmin değil ölçüm; ölçümlerin kendisi kodun
yanında ve [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) içinde.

### ✅ Kararlı

| Özellik | Not |
|---|---|
| **Rehberli telefon çekimi + 360 dikiş** | Telefon videosu → 8K'ya kadar equirectangular panorama. Döngü kapanışlı, yalnızca dönme modelli bundle adjustment; elde tutmanın paralaksına karşı mesh hizalama; graph-cut dikiş; çok bantlı harmanlama; kutup dolgusu. Referans gerçek çekim 4096 px'te **~170 sn**'de işleniyor (önceden 1278 sn), 16 thread'li bir dizüstünde. |
| **Sentetik kalite kapısı** | Prosedürel bir oda render eder, telefon çekimini simüle eder (hareket bulanıklığı, rolling shutter, pozlama kayması, gerçek paralaks) ve dikişi yer gerçeğiyle katı açı ağırlıklı PSNR/SSIM ile ölçer. |
| **360 oynatıcı** | Kapı eşiklerindeki zemin halkalarıyla odalar arası yürüme, oda çubuğu, kademeli çözünürlük, gömülebilir `<iframe>`. |
| **Ölçüm ve kat planı** | Zeminde mesafe ölçümü (ölçek kamera yüksekliğinden). Her panoramadan oturtulan oda, boyutları bilinen odalarda gerçek zeminle **0,984–0,991 IoU** örtüşüyor, kenarlar 9 cm içinde; plan elle düzenlenebiliyor. |
| **Hesaplar, QR ile çekim, yükleme** | Telefon QR'dan giriş yapmadan çekim yapar (tek eve bağlı, 12 saatlik, yalnızca çekim rotalarında geçerli bir anahtar). Parça parça, kaldığı yerden devam eden yükleme; canlı iş ilerlemesi. |

### 🔑 Anthropic API anahtarıyla çalışanlar

| Özellik | Not |
|---|---|
| **İlan metni** | Türk ilan portalları için başlık ve açıklama; metindeki her sayı ölçülen verilere karşı denetlenir. Claude Haiku ile ilan başına yaklaşık 0,005 $. |
| **Yatırım asistanı** | Oynatıcının içinde kira getirisi, geri ödeme süresi, alım maliyetleri, kredi ve bölgedeki fiyat eğilimi üzerine bir sohbet. Bütün hesapları motorun deterministik araçları yapar, piyasa verisi kaynak gösterilerek web aramasından gelir. Varsayılan model Claude Sonnet 5.5, soru başına yaklaşık 0,03 $; ilan başına ve toplam günlük bütçe sınırları var. Tahmin verir, yatırım tavsiyesi değil. |

### 🧪 Deneysel

| Özellik | Not |
|---|---|
| **VR** | Gözlüklerde WebXR (Quest, Vision Pro) ve telefonla "VR gözlük" (Cardboard) modu; mercek düzeltmesi ve bakarak seçme ile, hem 360 turda hem 3B gezintide. 104 birim testi ve masaüstü kontrolü (`?vr=cardboard`) var; **gerçek bir gözlükte ya da telefonda henüz denenmedi**. |
| **3B gezinti** | Bir yürüyüş videosundan Gaussian splat modeli: ALIKED + LightGlue, COLMAP/GLOMAP, Brush eğitimi, SOG çıktısı, PlayCanvas görüntüleyici. Dikkatle çekilmiş videoda çalışıyor; hızlı, bulanık ya da mesajlaşma uygulamasından geçmiş bir video ayrı parçalara bölünüyor. GPU ve yapı başına yaklaşık 30 dakika gerekiyor. |

### 🔭 Sıradaki adım

| Özellik | Not |
|---|---|
| **Mobilya kaldırma ve sanal dekorasyon** | Prototip olarak çalışıyor. Panoramada mobilyanın yerini odanın kendi duvar ve zeminiyle doldurmak, koyu leke oranını 0,43–0,71'den 0,08–0,11'e indirdi; ama yerel modellerle (LaMa, Stable Diffusion 1.5 + ControlNet, Mask2Former) üretim kalitesine ulaşmıyor: zeminler hâlâ düz kalıyor, dekorasyon odayı yeniden tasarlamak yerine mobilyanın yüzünü değiştiriyor. Bunun için daha büyük modeller ve ciddi bir GPU bütçesi gerekiyor; bu yüzden odağımı çekirdek 360 motoruna verdim — bu kısım sıradaki adım. |

---

## Proje hikâyesi

BioSphere'i emlakçılar için bir ürün olarak tasarladım ve her katmanını tek başıma geliştirdim:
dikiş motoru (hazır bir 360 kütüphanesine dayanmadan, sıfırdan), 3B yeniden yapılandırma, web
oynatıcı, panel ve yapay zekâ araçları.

Sonra emlakçılarla tek tek görüştüm. Çoğu ilan için video bile çekmiyor; bir dakikalık bir 360
çekimi de çoğu için ya fazladan bir zahmet ya da tercih etmedikleri bir şey. Büyük ilan
portalları da benzer bir turu üyelerine zaten ücretsiz sunuyor. Bu yüzden rafa kaldırmak yerine
açık kaynak yaptım; herkes erişebilsin, kullanabilsin, üzerine bir şey inşa edebilsin. Bir telefon
videosunu gezilebilir bir 360° tura çeviren sistemi baştan sona kendi başıma kurabilmiş olmak beni
gerçekten mutlu ediyor.

## Nasıl çalışıyor

```
360 tur
  telefon videosu ─► keskin anahtar kareler ─► öznitelik eşleştirme ─► dönüşler (bundle adjustment + döngü kapanışı)
                  ─► pozlama ve vinyet ─► mesh hizalama ─► graph-cut dikiş ─► çok bantlı harmanlama
                  ─► kutup dolgusu ─► 512 / 2048 / 4096 / 8192 px equirect ─► three.js oynatıcı ─► <iframe>

3B gezinti
  yürüyüş videosu ─► anahtar kareler ─► ALIKED + LightGlue ─► COLMAP / GLOMAP ─► düzleme ve ölçek
                  ─► odaların gezintiye kaydı (Sim3) ─► Brush 3DGS eğitimi ─► temizlik ─► .sog
                  ─► PlayCanvas görüntüleyici
```

Motor Python (FastAPI, OpenCV, NumPy/SciPy, SQLite); panel, çekim uygulaması ve oynatıcılar
Next.js 16 / React 19, three.js ve PlayCanvas. Tasarım kararları ve arkalarındaki ölçümler
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) içinde.

## Hızlı başlangıç

### Gereksinimler

| | |
|---|---|
| **FFmpeg** | 6+ (kare ayıklama ve kodlama) |
| **Python** | 3.11–3.13, [uv](https://docs.astral.sh/uv/) ile |
| **Node** | 22+ |
| **GPU** | İsteğe bağlı. CUDA varsa stüdyo modu RAFT optik akışını kullanır, yoksa CPU'da OpenCV DIS. 3B gezinti ve yapay zeka araçları için gerekli. |

Betikler, projenin geliştirildiği Windows (PowerShell) için yazıldı. Linux ve macOS'te aşağıdaki
elle kurulum adımları çalışır.

### Windows

```powershell
./scripts/setup.ps1            # CUDA PyTorch ve yapay zeka ekleri için -Gpu
./scripts/dev.ps1              # motor :8008'de, panel :3000'de
```

Ya da `scripts/start-dev.cmd` dosyasına çift tıklayın: PowerShell politikası gerektirmez; motoru,
paneli ve (sertifika ürettiyseniz) telefon vekilini her biri kendi penceresinde açar.

PowerShell yerel betikleri çalıştırmayı reddederse:
`powershell -ExecutionPolicy Bypass -File ./scripts/setup.ps1`.

### Elle kurulum

```bash
# motor
cd engine
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"            # ekler: gpu, ai, listing, walkthrough
cp .env.example .env                   # her anahtar isteğe bağlı
.venv/bin/python -m uvicorn biosphere.api.main:app --port 8008   # Windows: .venv\Scripts\python.exe

# web (ikinci terminal)
cd web
npm install
npm run dev
```

- Panel: <http://localhost:3000> — kendi bilgisayarınızda giriş istemez (`BIOSPHERE_AUTH_MODE=local`, varsayılan); başkalarının erişeceği bir kurulumda `accounts` yapın.
- Motor: <http://127.0.0.1:8008> · API dokümanı `/docs` adresinde.
- `GET /api/v1/health` bu makinenin neleri yapabildiğini söyler (FFmpeg, GPU, 3B gezinti araçları, yapay zeka modelleri).

### İsteğe bağlı özellikler

| Özellik | Gerekenler |
|---|---|
| İlan metni, yatırım asistanı | `uv pip install -e ".[listing]"`, ardından panelde **Ayarlar** (`/ayarlar`) sayfasına **kendi** Anthropic anahtarınızı yapıştırın. Proje anahtarla gelmez; `engine/.env` içinde `BIOSPHERE_LISTING_API_KEY` / `ANTHROPIC_API_KEY` de çalışır |
| 3B gezinti | `uv pip install -e ".[walkthrough]"` (pycolmap, kornia, LightGlue), [Brush](https://github.com/ArthurBrussee/brush) v0.3.0 ikilisi ve `@playcanvas/splat-transform` — `engine/.env.example` içindeki 3B gezinti bölümüne bakın |
| Mobilya kaldırma, dekorasyon | `-Gpu` kurulumu (CUDA PyTorch + `ai` eki); model ağırlıkları yalnızca `BIOSPHERE_AI_ALLOW_DOWNLOADS=true` ise ilk kullanımda iner |
| Yerel ağdaki telefonla çekim | Kamera ve hareket sensörleri HTTPS ister: yerel bir sertifika üretin (komut `scripts/certs/san.cnf` içinde), `node scripts/lan-proxy.mjs` çalıştırın ve telefonda `https://<bilgisayarın-IP'si>:3443` adresini açın |

## Kaliteyi ölçmek

Burada "görsel bozulma yok" bir görüş değil, bir ölçüm:

```bash
cd engine
.venv/bin/python -m pytest tests/test_pipeline_quality.py -q          # sentetik uçtan uca kapı
.venv/bin/python -m biosphere.synth.cli make-capture --style rich --trajectory three_row --effects realistic --out ./cap
```

## Geliştirme

```bash
cd engine && .venv/bin/python -m pytest -m "not slow"     # motor testleri (~560)
cd web && npx tsc --noEmit && npm test                    # tip denetimi + ~630 birim testi
```

## Herkese açık bir kurulum çalıştırırsanız

BioSphere her şeyi yerelde saklar (SQLite ve bir medya klasörü). Başkalarına hizmet verecekseniz
işleten sizsiniz: kendi gizlilik metniniz ve kullanım koşullarınız gerekir. İsteğe bağlı yapay
zeka özellikleri açıldığında veriyi Anthropic'e gönderir.

## Katkı

Issue ve pull request'lere açığız — [CONTRIBUTING.md](CONTRIBUTING.md). En çok fark yaratacak
yerler: daha iyi mobilya kaldırma modelleri, VR'ı gerçek gözlük ve telefonlarda denemek, 3B
gezinti için bir çekim rehberi, Linux/macOS kurulum betikleri ve bir Docker imajı. Güvenlik
açıkları için: [SECURITY.md](SECURITY.md).

## Teşekkürler

OpenCV, NumPy/SciPy, FastAPI, three.js, PlayCanvas ve `splat-transform`, Brush,
COLMAP/pycolmap/GLOMAP, LightGlue ve ALIKED, DINOv2, LaMa, Stable Diffusion, Mask2Former ve
Depth Anything gibi pek çok açık kaynak projenin emeği üzerine kurulu. Model ağırlıkları bu
depoda yok ve kendi lisanslarına tabi — bkz. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Lisans

[MIT](LICENSE) © 2026 Bilal Gürkan Şanlı ve BioSphere AI katkıcıları.
