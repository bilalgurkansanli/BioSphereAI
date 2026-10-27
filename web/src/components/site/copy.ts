/**
 * Every word on the public site, in both languages.
 *
 * Kept apart from the dashboard's dictionary because the two speak to different people: the
 * dashboard talks to whoever runs an installation, this page talks to somebody deciding
 * whether the repository is worth their afternoon. Structured rather than flat — the status
 * table, the pipeline and the roadmap are lists of things, and a list is easier to keep
 * honest when each entry carries both of its languages side by side.
 *
 * ── THE RULES THIS COPY FOLLOWS ──────────────────────────────────────────────────────
 * - Every number is a measurement that exists elsewhere in the repository (README, the
 *   architecture notes, the bench scripts). Nothing here is rounded up or "about" anything
 *   it was not measured to be.
 * - Limits are stated in the same breath as the feature. A status table that only has green
 *   rows is a brochure.
 * - No competitor is named, and nothing is sold. There is no company behind this page.
 */

import { CLONE_URL, REPO_NAME } from "@/lib/project";
import type { Locale } from "@/lib/types";

export interface Text {
  tr: string;
  en: string;
}

export const pick = (text: Text, locale: Locale): string => text[locale];

// ── Header and navigation ──────────────────────────────────────────────────────────────

export const NAV: { href: string; label: Text }[] = [
  { href: "/#durum", label: { tr: "Durum", en: "Status" } },
  { href: "/#nasil", label: { tr: "Nasıl çalışır", en: "How it works" } },
  { href: "/rehber", label: { tr: "Rehber", en: "Guide" } },
  { href: "/#katki", label: { tr: "Katkı", en: "Contribute" } },
];

export const HEADER = {
  home: { tr: "BioSphere AI — ana sayfa", en: "BioSphere AI — home" },
  github: { tr: "GitHub'da gör", en: "View on GitHub" },
  githubShort: { tr: "GitHub", en: "GitHub" },
  dashboard: { tr: "Paneli aç", en: "Open the dashboard" },
  language: { tr: "Dil", en: "Language" },
  nav: { tr: "Sayfa bölümleri", en: "Page sections" },
} satisfies Record<string, Text>;

// ── Hero ─────────────────────────────────────────────────────────────────────────────────

export const HERO = {
  eyebrow: { tr: "Kişisel proje · Açık kaynak (MIT)", en: "Personal project · Open source (MIT)" },
  titleLead: { tr: "Telefonla çekilen videodan", en: "From a phone video" },
  titleAccent: { tr: "360° sanal tur", en: "to a 360° tour" },
  titleTail: { tr: "— açık kaynak.", en: "— open source." },
  body: {
    tr: "Odanın ortasında telefonu bir tur çevirin: BioSphere en keskin kareleri seçer, kameranın nasıl döndüğünü çözer ve 8K'ya kadar küresel bir panorama diker. Dikiş motorunu hazır bir 360 kütüphanesine dayanmadan, sıfırdan kendim yazdım; odaları birbirine bağlayan oynatıcıyı, kat planını ve tura gömülen yapay zekâ araçlarını da.",
    en: "Turn your phone once around in the middle of a room: BioSphere picks the sharpest frames, solves how the camera rotated and stitches a spherical panorama of up to 8K. I wrote the stitching engine from scratch rather than on top of an off-the-shelf 360 library — and the player that links the rooms, the floor plan and the AI tools built into the tour.",
  },
  badges: {
    license: { tr: "MIT lisansı", en: "MIT licence" },
    local: { tr: "Kendi bilgisayarınızda çalışır", en: "Runs on your own machine" },
    stack: { tr: "Python + Next.js", en: "Python + Next.js" },
  },
  ctaPrimary: { tr: "GitHub'da gör", en: "View on GitHub" },
  ctaSecondary: { tr: "Kurulum rehberi", en: "Setup guide" },
  cloneLabel: { tr: "Depoyu klonlayın", en: "Clone the repository" },
} satisfies Record<string, Text | Record<string, Text>>;

export const ORB = {
  alt: {
    tr: "Motorun sentetik kalite testinde kullandığı prosedürel odanın 360° görüntüsü",
    en: "A 360° view of the procedural room the engine's synthetic quality gate uses",
  },
  badge: { tr: "Sentetik oda", en: "Synthetic room" },
  drag: { tr: "Sürükleyin", en: "Drag to look" },
  caption: {
    tr: "Bu oda gerçek değil: motorun kalite kapısının ürettiği prosedürel oda. Her dikiş, bu odanın kusursuz görüntüsüne karşı puanlanır.",
    en: "This room is not real: it is the procedural room the engine's quality gate renders. Every stitch is scored against its ground truth.",
  },
} satisfies Record<string, Text>;

// ── Screenshots ──────────────────────────────────────────────────────────────────────────

export const GALLERY = {
  eyebrow: { tr: "Ekran görüntüleri", en: "Screenshots" },
  title: { tr: "Kurduktan sonra göreceğiniz ekranlar", en: "What you get once it is running" },
  lead: {
    tr: "Görüntüler temiz bir kurulumda, sentetik odalarla alındı; hiçbirinde kimsenin evi yok.",
    en: "Captured on a clean install with synthetic rooms — nobody's home appears in any of them.",
  },
  tabs: { tr: "Ekran görüntüsü seçin", en: "Choose a screenshot" },
} satisfies Record<string, Text>;

export interface Shot {
  id: string;
  src: string;
  width: number;
  height: number;
  /** What the browser frame's address bar shows. Illustrative, never a real host. */
  path: string;
  label: Text;
  caption: Text;
  alt: Text;
}

export const SHOTS: Shot[] = [
  {
    id: "player",
    src: "/shots/player.webp",
    width: 1600,
    height: 1000,
    path: "localhost:3000/embed/…",
    label: { tr: "360° oynatıcı", en: "360° player" },
    caption: {
      tr: "Üstte oda çubuğu, kapı eşiğinde bir zemin halkası; altta kuş bakışı, ölçüm ve VR. Çözünürlük, keskin katmanlar geldikçe 8K'ya kadar kademeli yükselir.",
      en: "The room bar on top and a floor ring at the doorway; plan, measuring and VR along the bottom. Resolution steps up to 8K as the sharper levels arrive.",
    },
    alt: {
      tr: "Sentetik bir odanın 360° oynatıcıdaki görünümü: kapı eşiğinde bir zemin halkası, üstte oda çubuğu, altta araçlar",
      en: "A synthetic room in the 360° player: a floor ring at the doorway, the room bar on top and tools along the bottom",
    },
  },
  {
    id: "plan",
    src: "/shots/plan.webp",
    width: 1600,
    height: 1000,
    path: "localhost:3000/embed/…",
    label: { tr: "Kat planı", en: "Floor plan" },
    caption: {
      tr: "Odalar kendi panoramalarından ölçülür ve kuş bakışı bir plana oturur; ziyaretçi planda nerede durduğunu ve nereye baktığını görür. Yerleşim panelde elle düzeltilebilir.",
      en: "Each room is measured from its own panorama and laid out from above; visitors see where they stand and which way they face. The layout can be corrected by hand in the dashboard.",
    },
    alt: {
      tr: "360° oynatıcıda açık kuş bakışı kat planı; dört sentetik oda ve kapı halkaları",
      en: "The bird's-eye floor plan open in the 360° player: four synthetic rooms and their doorway rings",
    },
  },
  {
    id: "property",
    src: "/shots/property.webp",
    width: 1600,
    height: 1000,
    path: "localhost:3000/evler/…",
    label: { tr: "Ev sayfası", en: "Property page" },
    caption: {
      tr: "Sahneler, işlerin ilerlemesi, gömme kodu ve ilan metni aynı sayfada.",
      en: "Scenes, job progress, the embed code and the listing copy on one page.",
    },
    alt: {
      tr: "Paneldeki ev sayfası: sahne kartları ve araçlar",
      en: "The property page in the dashboard: scene cards and tools",
    },
  },
  {
    id: "capture",
    src: "/shots/capture.webp",
    width: 1600,
    height: 1000,
    path: "localhost:3000/evler/…",
    label: { tr: "Telefonla çekim", en: "Phone capture" },
    caption: {
      tr: "Masaüstünde bir QR kod. Telefon giriş yapmadan çeker; bağlantı 12 saat geçerli ve yalnızca o eve açık.",
      en: "A QR code on the desktop. The phone films without signing in; the link lasts 12 hours and opens only that property.",
    },
    alt: {
      tr: "“Sahne ekle” penceresinde telefonla çekim için QR kod",
      en: "The “Add scene” dialog with a QR code for filming on a phone",
    },
  },
  {
    id: "vr",
    src: "/shots/vr.webp",
    width: 1600,
    height: 1000,
    path: "localhost:3000/embed/…?vr=cardboard",
    label: { tr: "VR (Cardboard)", en: "VR (Cardboard)" },
    caption: {
      tr: "Telefonla VR kutusu modu: iki göz için ayrı görüntü, mercek düzeltmesi ve bakışla seçim. Deneysel — gerçek bir gözlükte henüz denenmedi.",
      en: "Phone VR-box mode: a view per eye, lens correction and gaze selection. Experimental — not yet tried in a real headset.",
    },
    alt: {
      tr: "Cardboard modunda iki göz için bölünmüş 360° görüntü",
      en: "The 360° view split for two eyes in Cardboard mode",
    },
  },
];

// ── Status ───────────────────────────────────────────────────────────────────────────────

export type StatusTone = "stable" | "key" | "experimental" | "parked";

export interface StatusItem {
  icon: "capture" | "player" | "plan" | "accounts" | "listing" | "invest" | "vr" | "walk" | "staging";
  title: Text;
  body: Text;
  facts: Text[];
}

export interface StatusGroup {
  tone: StatusTone;
  label: Text;
  blurb: Text;
  items: StatusItem[];
}

export const STATUS = {
  eyebrow: { tr: "Proje durumu", en: "Project status" },
  title: { tr: "Neler hazır, neler gelişiyor", en: "What is ready, and what is evolving" },
  lead: {
    tr: "Ekim 2026 itibarıyla. Her bileşeni ölçerek geliştirdim; sayılar tahmin değil ölçüm ve kodun yanında duruyor. Nerede sınıra geldiğimi de açıkça yazdım.",
    en: "As of October 2026. Every part was built by measuring it; the numbers are measured, not estimated, and they live next to the code. Where I hit a limit, I say so.",
  },
  summary: { tr: "Bileşenlerin durumu", en: "Component status" },
  signature: { tr: "— projenin geliştiricisi", en: "— the developer" },
} satisfies Record<string, Text>;

export const STATUS_GROUPS: StatusGroup[] = [
  {
    tone: "stable",
    label: { tr: "Kararlı", en: "Stable" },
    blurb: { tr: "İyi çalışıyor", en: "Works well" },
    items: [
      {
        icon: "capture",
        title: { tr: "Rehberli telefon çekimi ve 360° dikiş", en: "Guided phone capture and 360° stitching" },
        body: {
          tr: "Projenin kalbi, sıfırdan yazdığım dikiş motoru. OpenCV'nin hazır stitcher'ı yerine kendi hattı: telefon videosundan 8K'ya kadar eşdikdörtgen panorama. Döngü kapatmalı, yalnızca dönüşe dayalı demet ayarı; elde çekimin paralaksına karşı mesh hizalama; graph-cut dikişler, çok bantlı harmanlama ve kutup doldurma. Sentetik bir kalite kapısı prosedürel bir oda render eder, telefon çekimini taklit eder (hareket bulanıklığı, rolling shutter, pozlama kayması, paralaks) ve çıktıyı gerçeğe karşı puanlar.",
          en: "The heart of the project: a stitching engine I wrote from scratch, a pipeline of its own rather than OpenCV's stock stitcher. Phone video to an equirectangular panorama of up to 8K. Rotation-only bundle adjustment with loop closure, mesh alignment against hand-held parallax, graph-cut seams, multi-band blending and pole fill. A synthetic quality gate renders a procedural room, simulates a phone capture (motion blur, rolling shutter, exposure drift, parallax) and scores the output against ground truth.",
        },
        facts: [
          { tr: "4096 px ≈ 170 sn (önce 1278 sn)", en: "4096 px in ~170 s (was 1278 s)" },
          { tr: "16 iş parçacıklı dizüstü", en: "16-thread laptop" },
        ],
      },
      {
        icon: "player",
        title: { tr: "360° oynatıcı", en: "360° player" },
        body: {
          tr: "Odalar, kapı eşiklerinden geçiren zemin halkaları, oda çubuğu, ölçüm aracı (ölçek kamera yüksekliğinden gelir), kademeli çözünürlük ve her sayfaya giren bir iframe.",
          en: "Rooms, floor rings that walk you through doorways, a room bar, a measuring tool (scale from the camera height), progressive resolution and an embeddable iframe.",
        },
        facts: [
          { tr: "three.js", en: "three.js" },
          { tr: "<iframe>", en: "<iframe>" },
        ],
      },
      {
        icon: "plan",
        title: { tr: "Panoramalardan kat planı", en: "Floor plan from the panoramas" },
        body: {
          tr: "Her odanın panoramasından oturtulan oda, boyutu bilinen odalarda gerçek zeminle 0,984–0,991 IoU örtüşüyor; kenarlar 9 cm içinde. Plan elle düzenlenebilir.",
          en: "Rooms fitted from each panorama overlap the true floor at 0.984–0.991 IoU, with edges within 9 cm, on rooms of known size. The plan can be edited by hand.",
        },
        facts: [
          { tr: "IoU 0,984–0,991", en: "IoU 0.984–0.991" },
          { tr: "kenar ≤ 9 cm", en: "edges ≤ 9 cm" },
        ],
      },
      {
        icon: "accounts",
        title: { tr: "Hesaplar, QR çekim bağlantıları, yüklemeler", en: "Accounts, QR capture links, uploads" },
        body: {
          tr: "Telefon giriş yapmadan çeker: QR bağlantısı 12 saat geçerli, tek bir eve bağlı ve yalnızca izin listesindeki rotalarda geçiyor. Parçalı, kaldığı yerden devam eden yüklemeler; işin ilerlemesi canlı izleniyor.",
          en: "The phone films without signing in: the QR link lasts 12 hours, is tied to one property and is honoured only on allow-listed routes. Chunked, resumable uploads and live job progress.",
        },
        facts: [
          { tr: "12 sa · tek ev", en: "12 h · one property" },
          { tr: "devam ettirilebilir yükleme", en: "resumable uploads" },
        ],
      },
    ],
  },
  {
    tone: "key",
    label: { tr: "API anahtarıyla", en: "With an API key" },
    blurb: { tr: "Bir Anthropic anahtarı gerekir", en: "Needs an Anthropic key" },
    items: [
      {
        icon: "listing",
        title: { tr: "İlan metni", en: "Listing copy" },
        body: {
          tr: "Türk ilan portalları için başlık ve açıklama. Metindeki her sayı, ölçülmüş gerçeklere karşı denetleniyor.",
          en: "Title and description for Turkish listing portals. Every number in the copy is audited against the measured facts.",
        },
        facts: [
          { tr: "≈ 0,005 $ / ilan", en: "~$0.005 per listing" },
          { tr: "Claude Haiku", en: "Claude Haiku" },
        ],
      },
      {
        icon: "invest",
        title: { tr: "Oynatıcıda yatırım asistanı", en: "Investment assistant in the player" },
        body: {
          tr: "Kira getirisi, amortisman, alım masrafları, kredi taksiti ve bölgedeki fiyat eğilimi. Bütün aritmetik motorun deterministik araçlarında yapılıyor; piyasa verisi web aramasından, kaynağıyla geliyor. İlan başına ve genel günlük bütçe sınırları var. Tahmindir, yatırım tavsiyesi değildir.",
          en: "Rental yield, payback, purchase costs, mortgage and the area's price trend. All arithmetic runs in the engine's deterministic tools; market data comes from web search with cited sources. Per-listing and global daily budget caps. Estimates, not investment advice.",
        },
        facts: [
          { tr: "≈ 0,03 $ / soru", en: "~$0.03 per question" },
          { tr: "varsayılan Sonnet 5.5", en: "Sonnet 5.5 by default" },
        ],
      },
    ],
  },
  {
    tone: "experimental",
    label: { tr: "Deneysel", en: "Experimental" },
    blurb: { tr: "Çalışıyor, sınırlarıyla", en: "Works, within limits" },
    items: [
      {
        icon: "vr",
        title: { tr: "VR", en: "VR" },
        body: {
          tr: "Gözlüklerde WebXR (Quest, Vision Pro) ve telefonla “VR kutusu” (Cardboard) modu: mercek düzeltmesi ve bakışla seçim. Birim testleri var ve masaüstünde denendi; gerçek bir gözlükte ya da telefonda henüz denenmedi.",
          en: "WebXR on headsets (Quest, Vision Pro) and a phone “VR box” (Cardboard) mode with lens correction and gaze selection. Unit-tested and checked on desktop; not yet tried on a real headset or phone.",
        },
        facts: [
          { tr: "WebXR", en: "WebXR" },
          { tr: "gerçek cihazda denenmedi", en: "untested on real devices" },
        ],
      },
      {
        icon: "walk",
        title: { tr: "3B gezinti", en: "3D walkthrough" },
        body: {
          tr: "Bir gezinti videosundan Gaussian splat modeli: ALIKED + LightGlue, COLMAP/GLOMAP, Brush eğitimi, SOG dışa aktarımı, PlayCanvas görüntüleyici. Özenle çekilmiş videoda çalışıyor; hızlı, bulanık ya da mesajlaşma uygulamasından geçmiş bir telefon videosu ayrı parçalara bölünüyor.",
          en: "A Gaussian-splat model from a walk-through video: ALIKED + LightGlue, COLMAP/GLOMAP, Brush training, SOG export, PlayCanvas viewer. Works on carefully captured video; a fast, blurry or messenger-compressed phone video breaks into separate parts.",
        },
        facts: [
          { tr: "GPU gerekir", en: "needs a GPU" },
          { tr: "≈ 30 dk / model", en: "~30 min per build" },
        ],
      },
    ],
  },
  {
    tone: "parked",
    label: { tr: "Sıradaki adım", en: "Next step" },
    blurb: { tr: "Prototip; geliştirilmeye açık", en: "Prototype; open for improvement" },
    items: [
      {
        icon: "staging",
        title: { tr: "Mobilya kaldırma ve sanal dekorasyon", en: "Furniture removal and virtual staging" },
        body: {
          tr: "Prototip olarak çalışıyor: panoramada mobilyanın yerini odanın kendi duvar ve zeminiyle doldurmayı denedim ve koyu leke oranını 0,43–0,71'den 0,08–0,11'e indirdim. Ama yerel modellerle (LaMa, SD 1.5 + ControlNet, Mask2Former) üretim kalitesine ulaşmıyor; zeminler hâlâ düz kalıyor, dekorasyon odayı yeniden tasarlamak yerine mobilyanın yüzünü değiştiriyor. Bunun için daha büyük modeller ve ciddi bir GPU bütçesi gerekiyor. Bu yüzden odağımı projenin çekirdeği olan 360 motoruna verdim; bu kısım sıradaki adım.",
          en: "Works as a prototype: I tried refilling furniture in panoramas from the room's own walls and floor, and cut the dark-smear share from 0.43–0.71 to 0.08–0.11. With local models (LaMa, SD 1.5 + ControlNet, Mask2Former) it does not reach production quality — floors still come out flat, and staging re-skins the furniture rather than redesigning the room. That needs larger models and a real GPU budget, so I kept my focus on the core 360 engine; this part is the next step.",
        },
        facts: [
          { tr: "LaMa", en: "LaMa" },
          { tr: "SD 1.5 + ControlNet", en: "SD 1.5 + ControlNet" },
          { tr: "Mask2Former", en: "Mask2Former" },
        ],
      },
    ],
  },
];

// ── Why open source ──────────────────────────────────────────────────────────────────────

export const STORY = {
  eyebrow: { tr: "Hikâye", en: "Story" },
  title: { tr: "Proje hikâyesi", en: "The story" },
  paragraphs: [
    {
      tr: "BioSphere'i emlakçılar için bir ürün olarak tasarladım: fotoğrafçı ya da 360 kamera gerektirmeden, telefonla çekilen bir videodan gezilebilir bir tur. Fikirden çalışan sisteme kadar her katmanı — dikiş motoru, 3B yeniden yapılandırma, web oynatıcı, panel ve yapay zekâ araçları — tek başıma geliştirdim.",
      en: "I designed BioSphere as a product for estate agents: a walkable tour from a phone video, with no photographer and no 360 camera. From the idea to a working system I built every layer myself — the stitching engine, 3D reconstruction, the web player, the dashboard and the AI tools.",
    },
    {
      tr: "Sonra emlakçılarla tek tek görüştüm. Gördüğüm şu oldu: çoğu emlakçı ilan için video bile çekmiyor; bir dakikalık bir 360 çekimi de çoğu için ya fazladan bir zahmet ya da tercih etmedikleri bir şey. Büyük ilan portalları da benzer bir turu üyelerine zaten ücretsiz sunuyor. Bu tabloda BioSphere'in bir ürün olarak tutunmayacağını düşündüm — ve bunu erken görmek, sistemi kurmak kadar değerli bir ders oldu.",
      en: "Then I met estate agents one by one. What I found: most agents don't even shoot video for a listing, and a one-minute 360 capture is, for most of them, either extra effort or simply not their preference. Large listing portals already offer a similar tour to their members for free. With that picture I concluded BioSphere would not hold up as a product — and seeing that early was as valuable a lesson as building the system.",
    },
    {
      tr: "Bu yüzden rafa kaldırmak yerine açık kaynak yaptım; herkes erişebilsin, kullanabilsin, üzerine bir şey inşa edebilsin. Hesaplamalı fotoğrafçılık, 3B yeniden yapılandırma, gerçek zamanlı render ve bir LLM ajanı — hepsi tek bir dizüstünde çalışıyor. Bir telefon videosunu gezilebilir bir 360° tura çeviren sistemi baştan sona kendi başıma kurabilmiş olmak beni gerçekten mutlu ediyor. Okuyun, çalıştırın, geliştirin; katkılar memnuniyetle karşılanır.",
      en: "So instead of shelving it, I made it open source — for anyone to access, use and build on. Computational photography, 3D reconstruction, real-time rendering and an LLM agent, all running on one laptop. Having built, end to end and on my own, a system that turns a phone video into a walkable 360° tour genuinely makes me happy. Read it, run it, build on it; contributions are welcome.",
    },
  ] satisfies Text[],
  facts: [
    { tr: "Tek geliştirici: motor, web, panel ve yapay zekâ", en: "One developer: engine, web, dashboard and AI" },
    { tr: "MIT lisansı: kullanın, değiştirin, dağıtın", en: "MIT licence: use it, change it, share it" },
    { tr: "Ölçerek geliştirildi: her karar bir sayıya dayanıyor", en: "Built by measuring: every decision rests on a number" },
    {
      tr: "İsteğe bağlı yapay zekâ özellikleri dışında her şey yerelde",
      en: "Everything but the optional AI features runs locally",
    },
  ] satisfies Text[],
};

// ── How it works ─────────────────────────────────────────────────────────────────────────

export interface PipelineNode {
  title: Text;
  detail: Text;
  tech: string;
}

export const PIPELINE = {
  eyebrow: { tr: "Mimari", en: "Architecture" },
  title: { tr: "Nasıl çalışır", en: "How it works" },
  lead: {
    tr: "İki ayrı hat. 360° tur saf dönüşe dayanır: telefon tek bir noktanın çevresinde döndüğü sürece kameranın konumu değil, yalnızca yönü çözülür. 3B gezinti ise kameranın yürüdüğü yolu çözer ve sahneyi Gaussian splat'lerle yeniden kurar.",
    en: "Two pipelines. The 360° tour rests on pure rotation: as long as the phone turns about one point, only the camera's orientation is solved, not its position. The 3D walkthrough solves the path the camera walked and rebuilds the scene from Gaussian splats.",
  },
  lane360: { tr: "360° tur", en: "360° tour" },
  lane3d: { tr: "3B gezinti", en: "3D walkthrough" },
  laneGate: { tr: "Kalite kapısı", en: "Quality gate" },
  gate: {
    tr: "Prosedürel oda → simüle telefon çekimi (hareket bulanıklığı, rolling shutter, pozlama kayması, paralaks) → aynı hat → gerçeğe karşı PSNR / SSIM. “Görünür kusur yok” bir görüş değil, bir ölçüm.",
    en: "Procedural room → simulated phone capture (motion blur, rolling shutter, exposure drift, parallax) → the same pipeline → PSNR / SSIM against ground truth. “No visible artefacts” is a measurement, not an opinion.",
  },
  stackEngine: { tr: "Motor", en: "Engine" },
  stackWeb: { tr: "Web", en: "Web" },
  docs: { tr: "Mimari notları (Türkçe)", en: "Architecture notes (Turkish)" },
  experimental: { tr: "deneysel", en: "experimental" },
} satisfies Record<string, Text>;

export const LANE_360: PipelineNode[] = [
  {
    title: { tr: "Telefon videosu", en: "Phone video" },
    detail: { tr: "rehberli çekim, oda başına bir tur", en: "guided capture, one turn per room" },
    tech: "MP4 · MOV",
  },
  {
    title: { tr: "Anahtar kareler", en: "Keyframes" },
    detail: { tr: "en keskin kareler seçilir", en: "the sharpest frames are kept" },
    tech: "FFmpeg",
  },
  {
    title: { tr: "Dönüşler", en: "Rotations" },
    detail: { tr: "demet ayarı + döngü kapatma", en: "bundle adjustment + loop closure" },
    tech: "SciPy",
  },
  {
    title: { tr: "Dikiş ve harmanlama", en: "Seams and blending" },
    detail: { tr: "mesh hizalama, graph-cut, çok bantlı", en: "mesh alignment, graph-cut, multi-band" },
    tech: "OpenCV",
  },
  {
    title: { tr: "8K equirect", en: "8K equirect" },
    detail: { tr: "kutup doldurma, 512 → 8192 px", en: "pole fill, 512 → 8192 px" },
    tech: "JPEG",
  },
  {
    title: { tr: "Oynatıcı / iframe", en: "Player / iframe" },
    detail: { tr: "odalar, plan, ölçüm, VR", en: "rooms, plan, measuring, VR" },
    tech: "three.js",
  },
];

export const LANE_3D: PipelineNode[] = [
  {
    title: { tr: "Gezinti videosu", en: "Walk-through video" },
    detail: { tr: "dairede yavaşça yürüyerek", en: "walking slowly through the flat" },
    tech: "MP4",
  },
  {
    title: { tr: "Öznitelikler", en: "Features" },
    detail: { tr: "kareler arası eşleşmeler", en: "matches between frames" },
    tech: "ALIKED + LightGlue",
  },
  {
    title: { tr: "Kamera yolu", en: "Camera path" },
    detail: { tr: "konum, ölçek, odaların yerleşimi", en: "poses, scale, rooms registered" },
    tech: "COLMAP / GLOMAP",
  },
  {
    title: { tr: "Eğitim", en: "Training" },
    detail: { tr: "3B Gaussian splat", en: "3D Gaussian splats" },
    tech: "Brush",
  },
  {
    title: { tr: "Sıkıştırma", en: "Compression" },
    detail: { tr: "temizlik ve SOG", en: "clean-up and SOG" },
    tech: "splat-transform",
  },
  {
    title: { tr: "Görüntüleyici", en: "Viewer" },
    detail: { tr: "yürüyüş ve serbest uçuş", en: "walk and free flight" },
    tech: "PlayCanvas",
  },
];

export const STACK = {
  engine: "Python 3.11–3.13 · FastAPI · OpenCV · NumPy / SciPy · SQLite",
  web: "Next.js 16 · React 19 · three.js · PlayCanvas · Tailwind CSS",
};

// ── Engineering notes ───────────────────────────────────────────────────────────────────

export interface Metric {
  value: Text;
  unit?: Text;
  label: Text;
  note: Text;
}

export const HIGHLIGHTS = {
  eyebrow: { tr: "Mühendislik", en: "Engineering" },
  title: { tr: "Ölçülmüş sayılar", en: "Measured numbers" },
  lead: {
    tr: "Her sayının arkasında bir ölçüm betiği var; çoğu ARCHITECTURE.md'de, neyin denenip neden bırakıldığıyla birlikte yazılı.",
    en: "Every number has a bench script behind it; most are written up in ARCHITECTURE.md, together with what was tried and why it was dropped.",
  },
  speedTitle: { tr: "Referans gerçek çekim, 4096 px", en: "Reference real capture, 4096 px" },
  speedBefore: { tr: "önce", en: "before" },
  speedAfter: { tr: "şimdi", en: "now" },
  speedNote: {
    tr: "Aynı çekim, aynı 16 iş parçacıklı dizüstü: 1278 saniyeden yaklaşık 170 saniyeye.",
    en: "Same capture, same 16-thread laptop: from 1278 seconds to about 170.",
  },
  seconds: { tr: "sn", en: "s" },
  techniquesTitle: { tr: "Ana fikirler", en: "Key ideas" },
} satisfies Record<string, Text>;

export const METRICS: Metric[] = [
  {
    value: { tr: "0,984–0,991", en: "0.984–0.991" },
    label: { tr: "Kat planı IoU", en: "Floor-plan IoU" },
    note: {
      tr: "Panoramadan oturtulan odaların, boyutu bilinen odalarda gerçek zeminle örtüşmesi. Kenarlar 9 cm içinde.",
      en: "How the rooms fitted from a panorama overlap the true floor, on rooms of known size. Edges within 9 cm.",
    },
  },
  {
    value: { tr: "8K", en: "8K" },
    label: { tr: "Eşdikdörtgen çıktı", en: "Equirectangular output" },
    note: {
      tr: "Oynatıcı 512 px'lik önizlemeyle açılır, ardından 2K, 4K ve 8K katmanlarına geçer.",
      en: "The player opens on a 512 px preview, then steps through the 2K, 4K and 8K levels.",
    },
  },
  {
    value: { tr: "≈ 0,005 $", en: "~$0.005" },
    label: { tr: "İlan metni başına", en: "Per listing" },
    note: {
      tr: "Claude Haiku ile; metindeki her sayı ölçülmüş gerçeklere karşı denetlenir.",
      en: "On Claude Haiku; every number in the copy is audited against the measured facts.",
    },
  },
  {
    value: { tr: "≈ 0,03 $", en: "~$0.03" },
    label: { tr: "Yatırım sorusu başına", en: "Per investment question" },
    note: {
      tr: "Varsayılan Sonnet 5.5; ilan başına ve genel günlük bütçe sınırlarıyla.",
      en: "Sonnet 5.5 by default, with per-listing and global daily budget caps.",
    },
  },
  {
    value: { tr: "12 sa", en: "12 h" },
    label: { tr: "QR çekim bağlantısı", en: "QR capture link" },
    note: {
      tr: "Tek bir ev, izin listesindeki rotalar; telefon giriş yapmaz.",
      en: "One property, allow-listed routes; the phone never signs in.",
    },
  },
];

export const TECHNIQUES: { title: Text; body: Text }[] = [
  {
    title: { tr: "Saf dönüş modeli", en: "A pure-rotation model" },
    body: {
      tr: "Telefon tek bir noktada döndüğü sürece sahnenin derinliği gerekmez: demet ayarı yalnızca dönüşleri çözer, döngü kapatma turun sonunu başına bağlar.",
      en: "While the phone turns about one point, scene depth is not needed: bundle adjustment solves rotations only, and loop closure ties the end of the turn to its start.",
    },
  },
  {
    title: { tr: "Paralaksa karşı mesh hizalama", en: "Mesh alignment against parallax" },
    body: {
      tr: "Telefon merceğin değil kolun ucunda döner. Yakın eşyalarda kalan kaymayı yerel bir mesh deformasyonu emer.",
      en: "A phone turns at the end of an arm, not about its lens. A local mesh warp absorbs what that leaves on nearby objects.",
    },
  },
  {
    title: { tr: "Graph-cut dikiş, çok bantlı harmanlama", en: "Graph-cut seams, multi-band blending" },
    body: {
      tr: "Dikiş, görüntülerin en az ayrıştığı yerden geçer; harmanlama her frekans bandında ayrı yapılır, kutuplar ayrıca doldurulur.",
      en: "Seams run where the images disagree least; blending happens per frequency band, and the poles are filled separately.",
    },
  },
  {
    title: { tr: "Sentetik kalite kapısı", en: "A synthetic quality gate" },
    body: {
      tr: "Prosedürel bir oda, simüle edilmiş bir telefon çekimi ve gerçeğe karşı bir puan. Bir değişiklik çıktıyı kötüleştirirse testler kırmızıya döner.",
      en: "A procedural room, a simulated phone capture and a score against ground truth. A change that makes the output worse turns the tests red.",
    },
  },
];

// ── Quick start ──────────────────────────────────────────────────────────────────────────

export const QUICKSTART = {
  eyebrow: { tr: "Kurulum", en: "Setup" },
  title: { tr: "Hızlı başlangıç", en: "Quick start" },
  lead: {
    tr: "Proje Windows'ta geliştirildi ve betikler PowerShell. Linux ve macOS'ta elle kurulum adımları aynı sonucu verir.",
    en: "The project was developed on Windows and the scripts are PowerShell. On Linux and macOS the manual steps get you to the same place.",
  },
  requirements: { tr: "Gereksinimler", en: "Requirements" },
  colTool: { tr: "Araç", en: "Tool" },
  colVersion: { tr: "Sürüm", en: "Version" },
  tabScripts: { tr: "Windows (betikler)", en: "Windows (scripts)" },
  tabManual: { tr: "Elle kurulum", en: "Manual" },
  engine: { tr: "Motor", en: "Engine" },
  web: { tr: "Web (ikinci terminal)", en: "Web (second terminal)" },
  policy: {
    tr: "PowerShell yerel betikleri çalıştırmayı reddederse betiği şöyle başlatın:",
    en: "If PowerShell refuses to run local scripts, start the script like this:",
  },
  open: { tr: "Sonra açın", en: "Then open" },
  urlPanel: { tr: "Panel", en: "Dashboard" },
  urlEngine: { tr: "Motor", en: "Engine" },
  urlDocs: { tr: "API belgeleri", en: "API docs" },
  firstAccount: {
    tr: "İlk oluşturduğunuz hesap, kurulumdaki sahipsiz verileri üstlenir. Bütün API anahtarları isteğe bağlıdır.",
    en: "The first account you create adopts any data already on the installation. Every API key is optional.",
  },
  operator: {
    tr: "Herkese açık bir kurulum işletiyorsanız işletmeci sizsiniz: kendi gizlilik bildiriminiz ve koşullarınız sizin sorumluluğunuzda. İsteğe bağlı yapay zekâ özellikleri açıkken veriler Anthropic'e gider.",
    en: "If you run a public instance, you are its operator: your own privacy notice and terms are your responsibility. The optional AI features send data to Anthropic when they are switched on.",
  },
  copy: { tr: "Kopyala", en: "Copy" },
  copied: { tr: "Kopyalandı", en: "Copied" },
} satisfies Record<string, Text>;

export const REQUIREMENTS: { tool: string; version: Text; why: Text; optional?: boolean }[] = [
  {
    tool: "FFmpeg",
    version: { tr: "6+", en: "6+" },
    why: { tr: "Kare ayıklama ve kodlama", en: "Frame extraction and encoding" },
  },
  {
    tool: "Python",
    version: { tr: "3.11–3.13, uv ile", en: "3.11–3.13, with uv" },
    why: { tr: "Motor", en: "The engine" },
  },
  {
    tool: "Node.js",
    version: { tr: "22+", en: "22+" },
    why: { tr: "Panel, çekim uygulaması, oynatıcılar", en: "Dashboard, capture app, players" },
  },
  {
    tool: "NVIDIA GPU",
    version: { tr: "isteğe bağlı", en: "optional" },
    why: {
      tr: "Stüdyo modunda RAFT, 3B gezinti ve yapay zekâ araçları",
      en: "RAFT in studio mode, the 3D walkthrough and the AI tools",
    },
    optional: true,
  },
  {
    tool: "Brush · splat-transform · pycolmap + LightGlue",
    version: { tr: "isteğe bağlı", en: "optional" },
    why: { tr: "Yalnızca 3B gezinti için", en: "Only for the 3D walkthrough" },
    optional: true,
  },
];

/** Code is code in both languages; only the comments are translated. */
export const COMMANDS = {
  scripts: {
    tr: `git clone ${CLONE_URL}
cd ${REPO_NAME}
./scripts/setup.ps1     # -Gpu: CUDA PyTorch ve yapay zekâ ekleri
./scripts/dev.ps1       # motor :8008, panel :3000
# ya da: scripts/start-dev.cmd (çift tıklayın)`,
    en: `git clone ${CLONE_URL}
cd ${REPO_NAME}
./scripts/setup.ps1     # -Gpu: CUDA PyTorch and the AI extras
./scripts/dev.ps1       # engine on :8008, dashboard on :3000
# or: scripts/start-dev.cmd (double-click)`,
  },
  policy: {
    tr: "powershell -ExecutionPolicy Bypass -File ./scripts/setup.ps1",
    en: "powershell -ExecutionPolicy Bypass -File ./scripts/setup.ps1",
  },
  engine: {
    tr: `cd engine
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
cp .env.example .env    # bütün anahtarlar isteğe bağlı
.venv/bin/python -m uvicorn biosphere.api.main:app --port 8008
# Windows: .venv\\Scripts\\python.exe -m uvicorn biosphere.api.main:app --port 8008`,
    en: `cd engine
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
cp .env.example .env    # every key is optional
.venv/bin/python -m uvicorn biosphere.api.main:app --port 8008
# Windows: .venv\\Scripts\\python.exe -m uvicorn biosphere.api.main:app --port 8008`,
  },
  web: {
    tr: `cd web
npm install
npm run dev`,
    en: `cd web
npm install
npm run dev`,
  },
} satisfies Record<string, Text>;

export const URLS: { label: keyof typeof QUICKSTART; href: string }[] = [
  { label: "urlPanel", href: "http://localhost:3000" },
  { label: "urlEngine", href: "http://127.0.0.1:8008" },
  { label: "urlDocs", href: "http://127.0.0.1:8008/docs" },
];

// ── Contributing ─────────────────────────────────────────────────────────────────────────

export const CONTRIBUTE = {
  eyebrow: { tr: "Katkı", en: "Contributing" },
  title: { tr: "Yardım aranıyor", en: "Help wanted" },
  lead: {
    tr: "Issue'lar ve pull request'ler açık. En çok fark yaratacak yerler şunlar; başka bir fikriniz varsa önce bir issue açın, birlikte bakalım.",
    en: "Issues and pull requests are welcome. These are the places where help would matter most; if you have another idea, open an issue first and we will look at it together.",
  },
  guide: { tr: "Katkı rehberi", en: "Contributing guide" },
  issues: { tr: "Açık issue'lar", en: "Open issues" },
} satisfies Record<string, Text>;

export const IDEAS: { icon: "eraser" | "glasses" | "footprints" | "terminal" | "container" | "languages"; title: Text; body: Text; tag: string }[] = [
  {
    icon: "eraser",
    tag: "AI · GPU",
    title: { tr: "Daha iyi mobilya kaldırma modelleri", en: "Better furniture-removal models" },
    body: {
      tr: "Düz, grimsi zeminler ve kaçan küçük eşyalar. Daha iyi bir segmentasyon ve inpainting hattı ya da makul maliyetle çalışan büyük bir model.",
      en: "Flat, greyish floors and missed small items. A better segmentation and inpainting pipeline, or a large model that runs at a sane cost.",
    },
  },
  {
    icon: "glasses",
    tag: "VR",
    title: { tr: "Gerçek cihazda VR testi", en: "VR on real devices" },
    body: {
      tr: "Quest, Vision Pro ya da bir Cardboard ile denemek ve bulduklarınızı bir issue'ya yazmak — kod yazmadan da çok değerli.",
      en: "Trying it on a Quest, a Vision Pro or a Cardboard and writing up what you find — valuable even without writing code.",
    },
  },
  {
    icon: "footprints",
    tag: "Docs · 3D",
    title: { tr: "3B gezinti için çekim rehberi", en: "A capture guide for 3D walkthroughs" },
    body: {
      tr: "Hangi hız, ışık ve rota tek parça bir model veriyor? Ölçülerek yazılmış kısa bir rehber.",
      en: "Which speed, light and route give a model in one piece? A short guide, written from measurements.",
    },
  },
  {
    icon: "terminal",
    tag: "DX",
    title: { tr: "Linux / macOS kurulum betikleri", en: "Linux / macOS setup scripts" },
    body: {
      tr: "setup.ps1 ve dev.ps1'in bash karşılıkları; elle kurulum adımları zaten çalışıyor.",
      en: "Bash counterparts of setup.ps1 and dev.ps1; the manual steps already work.",
    },
  },
  {
    icon: "container",
    tag: "DX",
    title: { tr: "Docker", en: "Docker" },
    body: {
      tr: "Motoru ve paneli tek komutla ayağa kaldıran, GPU'lu ve GPU'suz çalışan bir imaj.",
      en: "An image that brings up the engine and the dashboard in one command, with and without a GPU.",
    },
  },
  {
    icon: "languages",
    tag: "Docs",
    title: { tr: "Mimari notlarının çevirisi", en: "Translating the architecture notes" },
    body: {
      tr: "ARCHITECTURE.md şimdilik Türkçe. İngilizce bir çeviri, projeyi okumak isteyen herkesin işine yarar.",
      en: "ARCHITECTURE.md is in Turkish for now. An English translation would help everyone who wants to read the project.",
    },
  },
];

// ── Footer ───────────────────────────────────────────────────────────────────────────────

export const FOOTER = {
  tagline: {
    tr: "Telefonla çekilen videodan 360° sanal tur. MIT lisanslı, açık kaynak bir öğrenme projesi.",
    en: "360° virtual tours from a phone video. An open-source, MIT-licensed learning project.",
  },
  project: { tr: "Proje", en: "Project" },
  page: { tr: "Bu sayfada", en: "On this page" },
  repo: { tr: "GitHub deposu", en: "GitHub repository" },
  readme: { tr: "README", en: "README" },
  architecture: { tr: "Mimari notları", en: "Architecture notes" },
  contributing: { tr: "Katkı rehberi", en: "Contributing" },
  license: { tr: "MIT lisansı", en: "MIT licence" },
  dashboard: { tr: "Paneli aç", en: "Open the dashboard" },
  operator: {
    tr: "Bu yazılımı herkese açık işleten kişi, kendi gizlilik bildiriminden sorumludur.",
    en: "Whoever runs a public instance of this software is responsible for their own privacy notice.",
  },
} satisfies Record<string, Text>;
