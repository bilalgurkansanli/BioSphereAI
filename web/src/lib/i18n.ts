/**
 * Minimal bilingual dictionary (Turkish default, English fallback).
 *
 * Deliberately not a full i18n framework: the surface is small, the keys are typed, and
 * a missing key is a compile error rather than a runtime "??key??" leaking into the UI.
 */

import type { JobStage, Locale, QualityMode } from "./types";

export const DEFAULT_LOCALE: Locale = "tr";

type Dict = Record<string, { tr: string; en: string }>;

export const strings = {
  "app.name": { tr: "BioSphere AI", en: "BioSphere AI" },
  "app.tagline": {
    tr: "Telefonla çek, 360° sanal tur al.",
    en: "Film with your phone, get a 360° tour.",
  },

  "nav.properties": { tr: "Evler", en: "Properties" },
  "nav.new": { tr: "Yeni Ev", en: "New Property" },
  "nav.settings": { tr: "Ayarlar", en: "Settings" },

  "property.empty.title": { tr: "Henüz eviniz yok", en: "No properties yet" },
  "property.empty.body": {
    tr: "İlk evinizi eklemek için bir oda videosu yükleyin.",
    en: "Upload a room video to add your first property.",
  },
  "property.create": { tr: "Ev Ekle", en: "Add Property" },
  "property.title": { tr: "Ev başlığı", en: "Property title" },
  "property.address": { tr: "Adres", en: "Address" },
  "property.scenes": { tr: "sahne", en: "scenes" },
  "property.views": { tr: "görüntülenme", en: "views" },
  "property.published": { tr: "Yayında", en: "Published" },
  "property.draft": { tr: "Taslak", en: "Draft" },
  "property.delete.confirm": {
    tr: "Bu ev ve bütün panoramaları kalıcı olarak silinecek. Emin misiniz?",
    en: "This property and all its panoramas will be permanently deleted. Are you sure?",
  },

  "scene.add": { tr: "Sahne Ekle", en: "Add Scene" },
  "scene.name": { tr: "Oda adı", en: "Room name" },
  "scene.reprocess": { tr: "Yeniden İşle", en: "Reprocess" },
  "scene.ready": { tr: "Hazır", en: "Ready" },
  "scene.processing": { tr: "İşleniyor", en: "Processing" },
  "scene.failed": { tr: "Başarısız", en: "Failed" },
  "scene.pending": { tr: "Sırada", en: "Queued" },

  "upload.drop": { tr: "Videoyu buraya sürükleyin", en: "Drop your video here" },
  "upload.or": { tr: "veya dosya seçin", en: "or choose a file" },
  "upload.hint": {
    tr: "MP4 / MOV / WEBM · en az 5 saniye · dosya başına en fazla 5 dakika ve 700 MB",
    en: "MP4 / MOV / WEBM · at least 5 seconds · up to 5 minutes and 700 MB per file",
  },
  "upload.photos": { tr: "Fotoğraf serisi yükle", en: "Upload photo series" },
  "upload.uploading": { tr: "Yükleniyor", en: "Uploading" },

  "capture.guide.title": { tr: "Nasıl çekilir?", en: "How to shoot" },
  "capture.guide.1": {
    tr: "Odanın ortasında durun ve telefonu dikey tutun.",
    en: "Stand in the middle of the room and hold the phone upright.",
  },
  "capture.guide.2": {
    tr: "Telefonu göz hizasında, kendi ekseninde yavaşça çevirin — gövdenizi değil.",
    en: "Turn the phone slowly around its own axis at eye level — not your body.",
  },
  "capture.guide.3": {
    tr: "Tam bir tur atın, sonra hafif yukarı ve hafif aşağı bakarak iki tur daha.",
    en: "Complete one full turn, then two more looking slightly up and slightly down.",
  },
  "capture.guide.4": {
    tr: "Bir turu yaklaşık 15 saniyede tamamlayın. Acele etmeyin.",
    en: "Take about 15 seconds per turn. Don't rush.",
  },

  "quality.fast": { tr: "Hızlı", en: "Fast" },
  "quality.balanced": { tr: "Dengeli", en: "Balanced" },
  "quality.studio": { tr: "Stüdyo", en: "Studio" },
  "quality.fast.desc": { tr: "4K · ~40 sn", en: "4K · ~40 s" },
  "quality.balanced.desc": { tr: "8K · ~2 dk", en: "8K · ~2 min" },
  "quality.studio.desc": { tr: "8K · ~6 dk · paralaks düzeltme", en: "8K · ~6 min · parallax repair" },

  "job.stage.upload": { tr: "Yükleniyor", en: "Uploading" },
  "job.stage.probe": { tr: "Video inceleniyor", en: "Inspecting video" },
  "job.stage.extract": { tr: "Kareler ayıklanıyor", en: "Extracting frames" },
  "job.stage.select": { tr: "En iyi kareler seçiliyor", en: "Selecting best frames" },
  "job.stage.features": { tr: "Görsel işaretler bulunuyor", en: "Finding visual features" },
  "job.stage.match": { tr: "Kareler eşleştiriliyor", en: "Matching frames" },
  "job.stage.bundle": { tr: "Kamera açıları çözülüyor", en: "Solving camera angles" },
  "job.stage.exposure": { tr: "Pozlama dengeleniyor", en: "Balancing exposure" },
  "job.stage.warp": { tr: "Küreye yansıtılıyor", en: "Projecting onto sphere" },
  "job.stage.seam": { tr: "Birleşim yerleri seçiliyor", en: "Choosing seams" },
  "job.stage.blend": { tr: "Görüntüler harmanlanıyor", en: "Blending images" },
  "job.stage.refine": { tr: "Paralaks düzeltiliyor", en: "Repairing parallax" },
  "job.stage.poles": { tr: "Tavan ve zemin tamamlanıyor", en: "Completing ceiling and floor" },
  "job.stage.levels": { tr: "Çözünürlükler hazırlanıyor", en: "Preparing resolutions" },
  "job.stage.done": { tr: "Tamamlandı", en: "Done" },
  "job.stage.sfm": { tr: "Kamera yolu çözülüyor", en: "Solving the camera path" },
  "job.stage.register": { tr: "Odalar daireye yerleştiriliyor", en: "Placing rooms in the flat" },
  "job.stage.train": { tr: "3B sahne eğitiliyor", en: "Training the 3D scene" },
  "job.stage.clean": { tr: "Artıklar temizleniyor", en: "Removing floaters" },
  "job.stage.export": { tr: "Sıkıştırılıyor, plan çiziliyor", en: "Compressing, drawing the plan" },

  "walk.title": { tr: "3B Gezinti", en: "3D Walkthrough" },
  "walk.loading": { tr: "3B sahne yükleniyor", en: "Loading the 3D scene" },
  "walk.mode.walk": { tr: "Yürüyüş", en: "Walk" },
  "walk.mode.drone": { tr: "Serbest uçuş", en: "Free flight" },
  "walk.autotour": { tr: "Otomatik tur", en: "Auto tour" },
  "walk.autotour.pause": { tr: "Turu duraklat", en: "Pause tour" },
  "walk.controls": { tr: "Kontroller", en: "Controls" },
  "walk.controls.move": { tr: "Hareket", en: "Move" },
  "walk.controls.look": { tr: "Bakış", en: "Look" },
  "walk.controls.zoom": { tr: "Yakınlaştır", en: "Zoom" },
  "walk.controls.walkTo": { tr: "Oraya yürü", en: "Walk to" },
  "walk.controls.touchLook": { tr: "Tek parmak: bak · İki parmak: yürü", en: "One finger: look · Two fingers: move" },
  "walk.controls.tapWalk": { tr: "Çift dokun: oraya yürü", en: "Double tap: walk there" },
  "walk.plan": { tr: "Kat planı", en: "Floor plan" },
  "walk.plan.hint": { tr: "Kat planı · girmek için bir noktaya tıklayın", en: "Floor plan · click a point to enter" },
  "walk.plan.close": { tr: "Planı kapat", en: "Close plan" },
  "walk.rooms": { tr: "Odalar", en: "Rooms" },
  "walk.general": { tr: "Genel gezinti", en: "Walkthrough" },
  "walk.unplaced": { tr: "Bu oda daireye yerleştirilemedi; ayrı bir alan olarak açılır.", en: "This room could not be placed in the flat; it opens as its own space." },
  "walk.unsupported": { tr: "Tarayıcınız 3B gezintiyi desteklemiyor (WebGL2 gerekli).", en: "Your browser cannot show the 3D walkthrough (WebGL2 needed)." },
  "walk.fullscreen": { tr: "Tam ekran", en: "Fullscreen" },
  "walk.to360": { tr: "360° görünüm", en: "360° view" },
  "walk.to3d": { tr: "3B gezinti", en: "3D walkthrough" },
  "walk.notReady": { tr: "3B gezinti henüz hazır değil.", en: "The 3D walkthrough is not ready yet." },

  "job.cancel": { tr: "İptal et", en: "Cancel" },
  "job.eta": { tr: "kalan süre", en: "remaining" },
  "job.retry": { tr: "Tekrar dene", en: "Retry" },

  "player.drag": { tr: "Bakmak için sürükleyin", en: "Drag to look around" },
  "player.vr": { tr: "VR gözlükle gez", en: "VR headset view" },
  "player.fullscreen": { tr: "Tam ekran", en: "Fullscreen" },
  // A verb pair, not a noun and a verb: the button sits in a row where every other label
  // says what pressing it does, and "Otomatik döndür" read as a heading next to them.
  "player.autorotate": { tr: "Döndürmeyi aç", en: "Start rotating" },
  // The only name this control has: the icon carries no word, so the accessible name has to
  // say which way the press goes rather than what the button is called.
  "player.autorotate.off": { tr: "Döndürmeyi durdur", en: "Stop rotating" },
  "player.rooms": { tr: "Odalar", en: "Rooms" },
  "player.loading": { tr: "Tur yükleniyor", en: "Loading the tour" },
  "player.quality": { tr: "Kalite", en: "Quality" },
  "player.empty": {
    tr: "Bu evde henüz görüntülenebilir bir oda yok.",
    en: "This property has no viewable rooms yet.",
  },
  "player.tools": { tr: "Araçlar", en: "Tools" },
  "player.empty.on": { tr: "Mobilyaları kaldır", en: "Remove furniture" },
  "player.empty.off": { tr: "Mobilyaları geri getir", en: "Bring the furniture back" },
  "player.empty.working": { tr: "Oda boşaltılıyor", en: "Emptying the room" },
  "player.empty.slow": {
    tr: "Oda boşaltılıyor. İlk seferde bir iki dakika sürebilir; sonra anında açılır.",
    en: "Emptying the room. The first time takes a minute or two; after that it opens instantly.",
  },
  "player.empty.staged": { tr: "Boş oda", en: "Empty room" },
  "player.empty.none": {
    tr: "Odadan ayrışan bir eşya bulunamadı.",
    en: "Nothing here separates from the room itself.",
  },
  "player.empty.failed": {
    tr: "Oda boşaltılamadı. Tekrar deneyin.",
    en: "The room could not be emptied. Try again.",
  },
  "player.portal.pending": {
    tr: "Bu odanın 360 çekimi henüz yapılmadı.",
    en: "This room has not been captured in 360 yet.",
  },
  "player.portal.back": { tr: "Geri dön", en: "Go back" },
  "player.portal.hint": {
    tr: "Zemindeki halkaya tıklayarak o odaya geçin.",
    en: "Click a ring on the floor to move to that room.",
  },
  "player.beta": {
    tr: "Bu araçlar beta aşamasında; sonuçlar her odada aynı kalitede olmayabilir.",
    en: "These tools are in beta; results are not equally good in every room.",
  },
  "player.decor": { tr: "Dekorasyon", en: "Staging" },
  "player.decor.on": { tr: "Dekore et", en: "Stage this room" },
  "player.decor.off": { tr: "Orijinale dön", en: "Back to the original" },
  "player.decor.working": { tr: "Oda döşeniyor", en: "Furnishing the room" },
  "player.decor.slow": {
    tr: "Bu birkaç dakika sürer; oda 16 pencereden tek tek döşeniyor.",
    en: "This takes a couple of minutes; the room is furnished a window at a time.",
  },
  "player.decor.failed": {
    tr: "Dekorasyon oluşturulamadı.",
    en: "The staging could not be generated.",
  },
  "player.decor.note": {
    tr: "Mobilya ve yüzeyler yapay zeka ile üretildi; odanın mimarisi korunur.",
    en: "Furniture and finishes are AI generated; the room's architecture is preserved.",
  },
  "player.empty.partial": {
    tr: "Duvarıyla aynı renkteki eşyalar kalabilir.",
    en: "Furnishings the colour of the surface behind them may remain.",
  },

  // The measure tool. Written long on purpose: every one of these sentences is the difference
  // between a number and a measurement, and the shorter versions all read as a promise the
  // geometry cannot keep. See `components/player/MeasureTool.tsx`.
  "player.measure": { tr: "Ölç", en: "Measure" },
  "player.measure.off": { tr: "Ölçmeyi bitir", en: "Stop measuring" },
  "player.measure.start": {
    tr: "Zeminde iki noktaya dokunun. Nesnenin zeminle birleştiği yeri işaretleyin.",
    en: "Tap two points on the floor. Mark where an object meets the floor.",
  },
  "player.measure.second": { tr: "İkinci noktaya dokunun.", en: "Tap the second point." },
  "player.measure.total": { tr: "toplam", en: "total" },
  "player.measure.undo": { tr: "Geri al", en: "Undo" },
  "player.measure.clear": { tr: "Temizle", en: "Clear" },
  "player.measure.height": { tr: "Kamera yüksekliği", en: "Camera height" },
  "player.measure.remeasure": { tr: "Yeniden ölç", en: "Measure again" },
  "player.measure.working": { tr: "Ölçülüyor", en: "Measuring" },
  "player.measure.failed": {
    tr: "Ölçüm yapılamadı. Tekrar deneyin.",
    en: "The measurement could not be made. Try again.",
  },
  "player.measure.scale.cameraHeight": {
    tr: "Ölçek tek bir sayıdan geliyor: kameranın zeminden {h} cm yükseklikte tutulduğu varsayımı. Bu yükseklik yanlışsa buradaki her ölçü aynı oranda yanlıştır.",
    en: "The scale comes from one number: the assumption that the camera was held {h} cm above the floor. If that height is wrong, every length here is wrong by the same proportion.",
  },
  "player.measure.scale.reference": {
    tr: "Ölçek, verilen bilinen uzunluktan geliyor.",
    en: "The scale comes from the known length that was supplied.",
  },
  "player.measure.scale.relative": {
    tr: "Metrik ölçek yok; bu sayılar yalnızca birbirlerine göre anlamlı.",
    en: "There is no metric scale; these numbers are only meaningful relative to each other.",
  },
  "player.measure.hint.horizon": {
    tr: "Bu nokta ufuk çizgisinde ya da üstünde; ışını zemine hiç ulaşmıyor. Nesnenin zeminle birleştiği yeri işaretleyin.",
    en: "This point is at or above the horizon, so its ray never reaches the floor. Mark the spot where the object meets the floor.",
  },
  "player.measure.hint.grazing": {
    tr: "Bu nokta neredeyse tam ufuk çizgisinde; zeminle kesiştiği yer anlamlı olamayacak kadar uzakta. Kameraya daha yakın bir nokta işaretleyin.",
    en: "This point is almost exactly on the horizon; the floor intersection is too far away to be meaningful. Mark a point closer to the camera.",
  },
  "player.measure.exact": {
    tr: "Bu araç yapay zeka kullanmaz: iki yön ve varsayılan bir yükseklikten kesin geometri.",
    en: "This tool uses no model: exact geometry from two directions and one assumed height.",
  },

  "plan.open": { tr: "Kuş bakışı", en: "Floor plan" },

  // The investment assistant — the player's "Yatırım" segment and the dashboard's panel. See
  // `components/player/InvestmentChat.tsx`. The disclosure is part of the panel itself, not a
  // footnote: the numbers are estimates, and saying so once, plainly, is what lets the answers
  // stay free of a disclaimer in every paragraph.
  "player.soon": { tr: "yakında", en: "soon" },
  "player.invest": { tr: "Yatırım", en: "Investment" },
  "player.invest.title": { tr: "Yatırım asistanı", en: "Investment assistant" },
  "player.invest.close": { tr: "Kapat", en: "Close" },
  "player.invest.body": {
    tr: "Kira getirisi, amortisman, bölgedeki fiyatlar ve kredi taksidi hakkında sorun. Rakamlar ilan bilgilerinden, açık kaynaklardan ve hesaplamadan gelir.",
    en: "Ask about rental yield, payback, prices in the area and mortgage payments. The figures come from the listing, public sources and calculation.",
  },
  "player.invest.note": {
    tr: "Tahmindir, yatırım tavsiyesi değildir. Gerçek getiri evin durumuna, kira sözleşmesine ve piyasaya göre değişir.",
    en: "These are estimates, not investment advice. The real return depends on the flat, the lease and the market.",
  },
  "player.invest.privacy": {
    tr: "Sorularınız yanıt üretmek için Anthropic'e (ABD) iletilir; kişisel bilgi yazmayın.",
    en: "Your questions are sent to Anthropic (USA) to generate answers; do not include personal details.",
  },
  "player.invest.loading": { tr: "Hesaplanıyor", en: "Calculating" },
  "player.invest.loadFailed": { tr: "Hesaplamalar yüklenemedi.", en: "The figures could not be loaded." },
  "player.invest.retry": { tr: "Tekrar dene", en: "Try again" },
  "player.invest.kpi.gross": { tr: "Brüt kira getirisi", en: "Gross rental yield" },
  "player.invest.kpi.payback": { tr: "Amortisman", en: "Payback" },
  "player.invest.kpi.perM2": { tr: "m² fiyatı", en: "Price per m²" },
  "player.invest.kpi.mortgage": { tr: "Örnek kredi taksidi", en: "Example mortgage" },
  "player.invest.kpi.net": { tr: "Net getiri", en: "Net yield" },
  "player.invest.kpi.costs": { tr: "Alım masrafları", en: "Purchase costs" },
  "player.invest.kpi.years": { tr: "yıl", en: "years" },
  "player.invest.kpi.perMonth": { tr: "/ay", en: "/mo" },
  "player.invest.kpi.rentExpected": { tr: "beklenen kira {rent}/ay ile", en: "at an expected rent of {rent}/mo" },
  "player.invest.kpi.rentActual": { tr: "mevcut kira {rent}/ay ile", en: "at the current rent of {rent}/mo" },
  "player.invest.kpi.noRent": { tr: "kira bilgisi girilmemiş", en: "no rent entered" },
  "player.invest.kpi.noArea": { tr: "alan girilmemiş", en: "no area entered" },
  "player.invest.kpi.paybackNote": { tr: "fiyat ÷ yıllık kira", en: "price ÷ yearly rent" },
  "player.invest.kpi.areaNet": { tr: "net {area} m²", en: "net {area} m²" },
  "player.invest.kpi.areaGross": { tr: "brüt {area} m²", en: "gross {area} m²" },
  "player.invest.kpi.mortgageNote": {
    tr: "{down} peşinat · {term} ay · aylık {rate} faiz",
    en: "{down} down · {term} months · {rate} a month",
  },
  "player.invest.kpi.netNote": {
    tr: "yılda {vacancy} ay boşluk, bakım, emlak vergisi, DASK ve alım masrafları sonrası",
    en: "after {vacancy} vacant month(s) a year, upkeep, property tax, DASK and purchase costs",
  },
  "player.invest.kpi.costsNote": {
    tr: "tapu harcı, döner sermaye, komisyon ve KDV, DASK",
    en: "title-deed fee, registry fee, commission with VAT, DASK",
  },
  "player.invest.kpi.rateSource": { tr: "Örnek faizin kaynağı", en: "Example rate source" },
  "player.invest.placeholder": { tr: "Bu ev hakkında bir soru sorun…", en: "Ask about this flat…" },
  "player.invest.send": { tr: "Gönder", en: "Send" },
  "player.invest.stop": { tr: "Durdur", en: "Stop" },
  "player.invest.newChat": { tr: "Yeni sohbet", en: "New conversation" },
  "player.invest.suggestions": { tr: "Örnek sorular", en: "Try asking" },
  "player.invest.status.thinking": { tr: "Düşünüyor…", en: "Thinking…" },
  "player.invest.status.searching": { tr: "Piyasa verileri aranıyor…", en: "Searching market data…" },
  "player.invest.status.searchingQuery": { tr: "“{query}” aranıyor…", en: "Searching “{query}”…" },
  "player.invest.status.computing": { tr: "Hesaplanıyor…", en: "Calculating…" },
  "player.invest.status.writing": { tr: "Yazıyor…", en: "Writing…" },
  "player.invest.sources": { tr: "Kaynaklar", en: "Sources" },
  "player.invest.calcs": { tr: "Hesaplama ayrıntısı", en: "Calculation details" },
  "player.invest.turnsLeft": { tr: "{n} soru hakkınız kaldı", en: "{n} question(s) left" },
  "player.invest.limitReached": {
    tr: "Bu sohbetin soru sınırı doldu. Yeni bir sohbet başlatabilirsiniz.",
    en: "This conversation has reached its question limit. You can start a new one.",
  },
  "player.invest.stopped": { tr: "Yanıt durduruldu.", en: "Answer stopped." },
  "player.invest.offline.no_key": {
    tr: "Sohbet bu sunucuda kapalı; yukarıdaki hesaplamalar ilan bilgileriyle günceldir.",
    en: "Chat is off on this server; the figures above are current with the listing.",
  },
  "player.invest.offline.budget": {
    tr: "Asistan bu ilan için bugünkü kullanım sınırına ulaştı. Yukarıdaki hesaplamalar geçerli; yarın tekrar sorabilirsiniz.",
    en: "The assistant has reached today's limit for this listing. The figures above stand; ask again tomorrow.",
  },
  "player.invest.retryIn": { tr: "{s} saniye sonra tekrar deneyin.", en: "Try again in {s} seconds." },

  // The agent's side: `components/dashboard/InvestmentPanel.tsx`.
  "invest.panel.title": { tr: "Yatırım asistanı", en: "Investment assistant" },
  "invest.panel.body": {
    tr: "Ziyaretçi turun içinden bu evin kira getirisini, amortismanını, bölgedeki fiyatları ve kredi taksidini sorar. Hesaplamaları motor yapar; piyasa verileri kaynağıyla birlikte web'den gelir. Ziyaretçi önce aşağıdaki kartı görür.",
    en: "Visitors ask, inside the tour, about this flat's yield, payback, prices in the area and mortgage payments. The engine does the arithmetic; market data comes from the web with its source. Visitors see the card below first.",
  },
  "invest.panel.enable": { tr: "Ziyaretçilere yatırım asistanını göster", en: "Offer the investment assistant to visitors" },
  "invest.panel.enable.desc": {
    tr: "Açıkken ilan bilgileriniz (fiyat, kira, konum — sokak adresi değil) ve ziyaretçinin soruları yanıt için Anthropic'e (ABD) iletilir.",
    en: "When on, your figures (price, rent, location — not the street address) and visitors' questions are sent to Anthropic (USA) to answer.",
  },
  "invest.panel.section.money": { tr: "Fiyat ve kira", en: "Price and rent" },
  "invest.panel.section.place": { tr: "Konum ve bina", en: "Location and building" },
  "invest.panel.section.costs": { tr: "Giderler ve kullanım", en: "Costs and usage" },
  "invest.field.price": { tr: "Satış fiyatı (TL)", en: "Asking price (TL)" },
  "invest.field.rent": { tr: "Aylık kira (TL)", en: "Monthly rent (TL)" },
  "invest.field.rentStatus": { tr: "Kira durumu", en: "Rent status" },
  "invest.field.rentStatus.expected": { tr: "Beklenen kira (ev boş)", en: "Expected rent (vacant)" },
  "invest.field.rentStatus.actual": { tr: "Mevcut kira (kiracılı)", en: "Current rent (let)" },
  "invest.field.aidat": { tr: "Aidat (TL/ay)", en: "Service charge (TL/month)" },
  "invest.field.emlak": { tr: "Emlak vergisi (TL/yıl)", en: "Property tax (TL/year)" },
  "invest.field.emlak.hint": {
    tr: "Boş bırakılırsa fiyat üzerinden üst sınır tahmini kullanılır.",
    en: "Left empty, an upper-bound estimate from the price is used.",
  },
  "invest.field.dask": { tr: "DASK (TL/yıl)", en: "Earthquake insurance (TL/year)" },
  "invest.field.il": { tr: "İl", en: "Province" },
  "invest.field.ilce": { tr: "İlçe", en: "District" },
  "invest.field.mahalle": { tr: "Mahalle", en: "Neighbourhood" },
  "invest.field.gross": { tr: "Brüt m²", en: "Gross m²" },
  "invest.field.net": { tr: "Net m²", en: "Net m²" },
  "invest.field.rooms": { tr: "Oda sayısı", en: "Rooms" },
  "invest.field.age": { tr: "Bina yaşı", en: "Building age" },
  "invest.field.floor": { tr: "Kat", en: "Floor" },
  "invest.field.totalFloors": { tr: "Toplam kat", en: "Floors in building" },
  "invest.field.heating": { tr: "Isınma", en: "Heating" },
  "invest.field.energy": { tr: "Enerji sınıfı", en: "Energy class" },
  "invest.field.saleKind": { tr: "Satış türü", en: "Sale type" },
  "invest.field.saleKind.second_hand": { tr: "İkinci el", en: "Second-hand" },
  "invest.field.saleKind.new_build": { tr: "Sıfır (yeni)", en: "New build" },
  "invest.field.site": { tr: "Site içinde", en: "In a gated complex" },
  "invest.field.yes": { tr: "Evet", en: "Yes" },
  "invest.field.no": { tr: "Hayır", en: "No" },
  "invest.field.unknown": { tr: "Belirtilmedi", en: "Not stated" },
  "invest.field.commission": { tr: "Alıcıdan komisyon (%, KDV hariç)", en: "Buyer commission (%, before VAT)" },
  "invest.field.commission.hint": {
    tr: "Boş bırakılırsa yasal tavan (%2) kullanılır; komisyon almıyorsanız 0 yazın.",
    en: "Left empty, the legal ceiling (2%) is used; enter 0 if you charge none.",
  },
  "invest.field.notes": { tr: "Asistan için notlar", en: "Notes for the assistant" },
  "invest.field.notes.placeholder": {
    tr: "Ör. metroya 5 dakika, kapalı otopark var, 2024'te tadilat görmüş",
    en: "e.g. 5 minutes to the metro, covered parking, renovated in 2024",
  },
  "invest.field.notes.hint": {
    tr: "Sizin beyanınız olarak kullanılır; ziyaretçiye olduğu gibi gösterilmez.",
    en: "Used as your statement; not shown to visitors verbatim.",
  },
  "invest.field.budget": { tr: "Günlük harcama sınırı (USD)", en: "Daily spending cap (USD)" },
  "invest.field.budget.hint": { tr: "Boş bırakılırsa {d} USD; en fazla {max} USD.", en: "Empty means {d} USD; at most {max} USD." },
  "invest.panel.save": { tr: "Kaydet", en: "Save" },
  "invest.panel.saved": { tr: "Yatırım bilgileri kaydedildi", en: "Investment details saved" },
  "invest.panel.failed": { tr: "Yatırım bilgileri kaydedilemedi", en: "Investment details could not be saved" },
  "invest.panel.preview": { tr: "Ziyaretçinin göreceği kart", en: "The card visitors see" },
  "invest.panel.previewEmpty": { tr: "Kartın çıkması için satış fiyatını girin.", en: "Enter the asking price to see the card." },
  "invest.panel.openEmbed": { tr: "Ziyaretçi görünümünü aç", en: "Open the visitor view" },
  "invest.panel.visible": { tr: "Ziyaretçilere açık", en: "Live for visitors" },
  "invest.panel.hidden": { tr: "Ziyaretçilere kapalı", en: "Not shown to visitors" },
  "invest.blocker.not_enabled": { tr: "Asistan bu ev için açık değil.", en: "The assistant is off for this property." },
  "invest.blocker.no_price": { tr: "Satış fiyatı girilmedi.", en: "No asking price entered." },
  "invest.blocker.not_published": { tr: "Ev yayında değil; ziyaretçiler göremez.", en: "The property is not published." },
  "invest.blocker.disabled": { tr: "Asistan bu sunucuda kapatılmış.", en: "The assistant is disabled on this server." },
  "invest.blocker.no_key": {
    tr: "Sunucuda yapay zeka anahtarı yok; ziyaretçi yalnızca hesaplama kartını görür.",
    en: "No AI key on the server; visitors see the figures card only.",
  },
  "invest.blocker.no_sdk": { tr: "Yapay zeka istemcisi kurulu değil.", en: "The AI client library is not installed." },
  "invest.blocker.budget": {
    tr: "Bugünkü harcama sınırına ulaşıldı; sohbet yarın yeniden açılır.",
    en: "Today's spending cap is reached; the chat reopens tomorrow.",
  },
  "invest.usage.title": { tr: "Kullanım ve maliyet", en: "Usage and cost" },
  "invest.usage.today": { tr: "Bugün", en: "Today" },
  "invest.usage.week": { tr: "Son 7 gün", en: "Last 7 days" },
  "invest.usage.month": { tr: "Son 30 gün", en: "Last 30 days" },
  "invest.usage.questions": { tr: "Soru (30 gün)", en: "Questions (30 days)" },
  "invest.usage.perQuestion": { tr: "Soru başına ortalama", en: "Average per question" },
  "invest.usage.searches": { tr: "Web araması (30 gün)", en: "Web searches (30 days)" },
  "invest.usage.cap": { tr: "Bugün {spent} / sınır {cap}", en: "Today {spent} / cap {cap}" },
  "invest.usage.model": { tr: "Model", en: "Model" },
  "invest.usage.none": { tr: "Henüz soru sorulmadı.", en: "No questions yet." },
  "invest.rates.title": { tr: "Hesaplamada kullanılan oranlar", en: "Rates the calculations use" },
  "invest.rates.asOf": { tr: "{date} tarihinde okundu", en: "read on {date}" },
  "invest.rates.stale": { tr: "bir yıldan eski", en: "older than a year" },
  "plan.title": { tr: "Kuş bakışı", en: "Floor plan" },
  "plan.close": { tr: "Planı kapat", en: "Close the plan" },
  "plan.scaleNote": {
    tr: "Ölçek, kameranın {h} m yükseklikte tutulduğu varsayımından geliyor.",
    en: "The scale follows from assuming the camera was held at {h} m.",
  },
  "plan.notShot": { tr: "çekilmedi", en: "not shot yet" },
  "plan.unlinked": { tr: "Henüz bir kapıyla bağlanmadı:", en: "No doorway reaches these yet:" },
  "plan.measure": { tr: "Kat planını çıkar", en: "Measure the floor plan" },
  "plan.measureMore": { tr: "{n} odayı daha ölç", en: "Measure {n} more room(s)" },
  "plan.measuring": { tr: "Odalar ölçülüyor", en: "Measuring the rooms" },
  "plan.noModel": {
    tr: "Bu sunucuda ölçüm modeli yok",
    en: "This deployment has no segmenter",
  },
  "plan.empty": {
    tr: "Odalar henüz ölçülmedi. Ölçüm, her odanın panoramasından zemin–duvar hattını okur.",
    en: "The rooms have not been measured yet. Measuring reads the floor-to-wall line off each room's panorama.",
  },
  "plan.failed": {
    tr: "Plan çıkarılamadı. Panoramalar hazır olduğunda tekrar deneyin.",
    en: "The plan could not be built. Try again once the panoramas are ready.",
  },
  "plan.summary": {
    tr: "{n} oda ölçüldü · toplam {area} m²",
    en: "{n} rooms measured · {area} m² in total",
  },
  "plan.edit": { tr: "Planı düzenle", en: "Edit the plan" },
  "plan.editDone": { tr: "Düzenlemeyi bitir", en: "Done editing" },
  "plan.editHint": {
    tr: "Odayı sürükleyin — duvarlar birbirine yaklaşınca yapışır. Seçili odayı çeyrek tur çevirebilirsiniz.",
    en: "Drag a room — walls snap flush when they come close. A selected room turns a quarter at a time.",
  },
  "plan.rotate": { tr: "Çeyrek tur çevir", en: "Turn a quarter" },
  "plan.resetRoom": { tr: "Otomatiğe dön", en: "Back to automatic" },
  "plan.resetAll": { tr: "Tüm düzeltmeleri geri al", en: "Undo every correction" },
  "plan.addRoom": { tr: "Plana ekle", en: "Add to the plan" },
  "plan.placedCount": {
    tr: "{n} oda elle yerleştirildi",
    en: "{n} room(s) placed by hand",
  },
  "plan.saveFailed": {
    tr: "Değişiklik kaydedilemedi.",
    en: "That change could not be saved.",
  },
  "plan.needsPortals": {
    tr: "Odaları birbirine bağlamak için kapı eşiklerine halka bırakın; plan halkalardan kuruluyor.",
    en: "Drop a floor ring in each doorway to link the rooms; the plan is built from them.",
  },
  // VR: a headset over WebXR, or a phone in a VR box ("Cardboard"). See `components/vr/`.
  // The dock segment's caption is "VR" and its full name is `player.vr`, so a screen reader
  // hears what the button does, not two letters.
  "vr.caption": { tr: "VR", en: "VR" },
  "vr.exit": { tr: "VR'dan çık", en: "Exit VR" },
  "vr.denied": {
    tr: "Hareket sensörü izni verilmedi. Sekmeyi kapatıp yeniden açın; izin sorulduğunda “İzin Ver”e dokunun.",
    en: "Motion sensor access was not granted. Close and reopen the tab, then choose “Allow” when asked.",
  },
  "vr.noSensor": {
    tr: "Bu cihazda hareket sensörü bulunamadı; bakmak için ekranı sürükleyin.",
    en: "This device reports no motion sensor; drag the screen to look around.",
  },
  "vr.failed": {
    tr: "VR başlatılamadı. Gözlüğün açık ve bağlı olduğundan emin olup tekrar deneyin.",
    en: "VR could not start. Make sure the headset is on and connected, then try again.",
  },
  "vr.pending": { tr: "Henüz çekilmedi", en: "Not shot yet" },
  "vr.card.close": { tr: "Kapatmak için yeniden seçin", en: "Select again to close" },
  "vr.guide.title": { tr: "VR gözlükle gezin", en: "Tour it in a VR headset" },
  "vr.guide.step1": {
    tr: "Telefonu yan çevirip gözlüğe yerleştirin; ekranın ortası iki merceğin tam ortasına gelsin.",
    en: "Turn the phone on its side and put it in the headset, with the middle of the screen between the two lenses.",
  },
  "vr.guide.step2": {
    tr: "Etrafınıza bakın. Zemindeki bir halkaya ya da bir etikete bakıp bekleyin; çember dolunca oraya geçersiniz.",
    en: "Look around. Rest your gaze on a ring on the floor or on a tag; when the circle fills, you go there.",
  },
  "vr.guide.step3": {
    tr: "Çıkmak için ekrana dokunun, geri tuşuna basın ya da ayağınızın dibindeki menüde “VR'dan çık”a bakın.",
    en: "To leave, tap the screen, press back, or look at “Exit VR” in the menu at your feet.",
  },
  "vr.guide.walkStep2": {
    tr: "Etrafınıza bakın. Zemindeki bir noktaya bakıp bekleyin; çember dolunca oraya geçersiniz.",
    en: "Look around. Rest your gaze on a spot on the floor; when the circle fills, you move there.",
  },
  "vr.guide.permission": {
    tr: "Başla'ya dokunduğunuzda telefon hareket sensörü için izin isteyecek; “İzin Ver”i seçin.",
    en: "When you tap Start, the phone will ask for motion-sensor access; choose “Allow”.",
  },
  "vr.guide.lens": { tr: "Gözlük", en: "Headset" },
  "vr.guide.cardboard": { tr: "Cardboard", en: "Cardboard" },
  "vr.guide.vrbox": { tr: "VR Box / diğer", en: "VR Box / other" },
  "vr.guide.start": { tr: "Başla", en: "Start" },
  "vr.guide.cancel": { tr: "Vazgeç", en: "Cancel" },
  "vr.starting": { tr: "Hazırlanıyor…", en: "Getting ready…" },
  "vr.rotate.title": { tr: "Telefonu yan çevirin", en: "Turn the phone on its side" },
  "vr.rotate.body": {
    tr: "Görüntü, telefon yan yatınca iki göz için ikiye bölünür.",
    en: "The picture splits in two, one half per eye, once the phone is on its side.",
  },
  "vr.handoff.title": { tr: "Bu pencerede VR açılamıyor", en: "VR cannot start in this window" },
  "vr.handoff.body": {
    tr: "Tur başka bir sitenin içinde gösterildiği için tarayıcı burada VR'a izin vermiyor. Aynı odayı yeni sekmede tam ekran açabilirsiniz.",
    en: "The tour is shown inside another site, where the browser does not allow VR. You can open the same room full screen in a new tab.",
  },
  "vr.handoff.open": { tr: "Yeni sekmede aç", en: "Open in a new tab" },
  "vr.prompt.title": { tr: "VR'a geçmeye hazır", en: "Ready for VR" },
  "vr.prompt.start": { tr: "VR'a geç", en: "Enter VR" },
  "vr.prompt.later": { tr: "Şimdi değil", en: "Not now" },
  "vr.walk.smooth": { tr: "Akıcı hareket", en: "Smooth moving" },
  "vr.walk.unavailable": {
    tr: "Bu tarayıcıda 3B gezinti VR'da açılamıyor; 360° görünümü VR'da deneyin.",
    en: "This browser cannot show the 3D walkthrough in VR; try the 360° view in VR instead.",
  },

  "editor.hotspots": { tr: "Bilgi Etiketleri", en: "Info Tags" },
  "editor.hotspot.add": { tr: "Etiket eklemek için panoramaya tıklayın", en: "Click the panorama to add a tag" },
  "editor.hotspot.title": { tr: "Etiket başlığı", en: "Tag title" },
  "editor.hotspot.body": { tr: "Açıklama", en: "Description" },
  "editor.hotspot.link": { tr: "Odaya geçiş", en: "Room link" },
  "editor.initialView": { tr: "Açılış görünümü olarak ayarla", en: "Set as opening view" },
  "editor.save": { tr: "Kaydet", en: "Save" },
  "editor.delete": { tr: "Sil", en: "Delete" },
  "editor.cancel": { tr: "Vazgeç", en: "Cancel" },

  "embed.title": { tr: "Siteye Göm", en: "Embed" },
  "embed.copy": { tr: "Kodu kopyala", en: "Copy code" },
  "embed.copied": { tr: "Kopyalandı", en: "Copied" },
  "embed.hint": {
    tr: "Bu kodu emlak portalınızın ilan açıklamasına yapıştırın.",
    en: "Paste this into your listing portal's description.",
  },

  "coverage.title": { tr: "Kapsama", en: "Coverage" },
  "coverage.complete": { tr: "Tam küre kapsandı", en: "Full sphere covered" },
  "coverage.gaps": { tr: "eksik bölge", en: "missing area" },
  "coverage.reshoot": {
    tr: "Turu bu bölgeleri de kapsayacak şekilde tekrar çekerseniz kalite artar.",
    en: "Re-shoot covering these areas to improve quality.",
  },

  "metrics.title": { tr: "Teknik Rapor", en: "Technical Report" },
  "metrics.keyframes": { tr: "Anahtar kare", en: "Keyframes" },
  "metrics.reprojection": { tr: "Hizalama hatası", en: "Alignment error" },
  "metrics.loop": { tr: "Tur kapanma hatası", en: "Loop closure error" },
  "metrics.hfov": { tr: "Görüş açısı", en: "Field of view" },
  "metrics.seam": { tr: "Dikiş görünürlüğü", en: "Seam visibility" },

  "error.generic": { tr: "Bir şeyler ters gitti.", en: "Something went wrong." },
  "error.offline": {
    tr: "Motor çalışmıyor. Terminalden `biosphere serve` çalıştırın.",
    en: "The engine is offline. Run `biosphere serve` in a terminal.",
  },
} satisfies Dict;

export type StringKey = keyof typeof strings;

export function t(key: StringKey, locale: Locale = DEFAULT_LOCALE): string {
  return strings[key][locale];
}

export function stageLabel(stage: JobStage, locale: Locale = DEFAULT_LOCALE): string {
  const key = `job.stage.${stage}` as StringKey;
  return key in strings ? t(key, locale) : stage;
}

export function qualityLabel(mode: QualityMode, locale: Locale = DEFAULT_LOCALE): string {
  return t(`quality.${mode}` as StringKey, locale);
}

export function qualityDescription(mode: QualityMode, locale: Locale = DEFAULT_LOCALE): string {
  return t(`quality.${mode}.desc` as StringKey, locale);
}
