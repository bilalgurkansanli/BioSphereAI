/**
 * Dashboard-local copy.
 *
 * `lib/i18n.ts` is frozen and owns every string the product spec named. The dashboard
 * needs a few dozen more — placeholders, confirmations, metric explanations — so they live
 * here, in the same shape, and are reported upstream as candidates for the shared
 * dictionary rather than smuggled into it.
 */

import type { Locale } from "@/lib/types";

type Dict = Record<string, { tr: string; en: string }>;

export const uiStrings = {
  // ── Shell ──────────────────────────────────────────────────────────────────────────
  // ── Accounts ───────────────────────────────────────────────────────────────────────
  "auth.signIn": { tr: "Giriş yap", en: "Sign in" },
  "auth.signUp": { tr: "Hesap oluştur", en: "Create an account" },
  "auth.signOut": { tr: "Çıkış yap", en: "Sign out" },
  "auth.signIn.title": { tr: "Tekrar hoş geldiniz", en: "Welcome back" },
  "auth.signIn.body": { tr: "Evlerinize devam etmek için giriş yapın.", en: "Sign in to get back to your properties." },
  "auth.signUp.title": { tr: "Hesabınızı oluşturun", en: "Create your account" },
  "auth.signUp.body": {
    tr: "Bu kurulumdaki hesabınız; bilgileriniz yalnızca bu sunucuda tutulur.",
    en: "An account on this installation; your details stay on this server.",
  },
  "auth.email": { tr: "E-posta", en: "E-mail" },
  "auth.name": { tr: "Ad soyad", en: "Full name" },
  "auth.password": { tr: "Şifre", en: "Password" },
  "auth.password.hint": { tr: "En az 10 karakter.", en: "At least 10 characters." },
  "auth.password.show": { tr: "Şifreyi göster", en: "Show password" },
  "auth.password.hide": { tr: "Şifreyi gizle", en: "Hide password" },
  "auth.haveAccount": { tr: "Zaten hesabınız var mı?", en: "Already have an account?" },
  "auth.noAccount": { tr: "Hesabınız yok mu?", en: "No account yet?" },
  "auth.or": { tr: "veya", en: "or" },
  "auth.google": { tr: "Google ile devam et", en: "Continue with Google" },
  "auth.google.failed": {
    tr: "Google ile giriş tamamlanamadı. Tekrar deneyin ya da e-posta ile girin.",
    en: "Google sign-in could not be completed. Try again, or use your e-mail.",
  },
  "auth.google.cancelled": { tr: "Google ile giriş yarıda bırakıldı.", en: "Google sign-in was cancelled." },
  "auth.google.closed": {
    tr: "Bu kurulumda yeni kayıt kapalı. Hesabınız varsa e-posta ile girebilirsiniz.",
    en: "New accounts are closed on this installation. Sign in with your e-mail if you have one.",
  },
  "auth.checking": { tr: "Oturum kontrol ediliyor", en: "Checking your session" },
  "auth.expired": { tr: "Oturumunuzun süresi doldu. Lütfen tekrar giriş yapın.", en: "Your session expired. Please sign in again." },
  "auth.account": { tr: "Hesap", en: "Account" },
  "auth.backHome": { tr: "Ana sayfaya dön", en: "Back to the home page" },

  "chrome.home": { tr: "Panel", en: "Dashboard" },
  "chrome.menu": { tr: "Menü", en: "Menu" },
  "chrome.locale": { tr: "Dil", en: "Language" },
  "chrome.breadcrumb": { tr: "Konum", en: "Breadcrumb" },
  "chrome.engine": { tr: "Motor durumu", en: "Engine status" },
  "chrome.engine.checking": { tr: "Kontrol ediliyor", en: "Checking" },
  "chrome.engine.online": { tr: "Motor çalışıyor", en: "Engine online" },
  "chrome.engine.degraded": { tr: "Motor uyarı veriyor", en: "Engine degraded" },
  "chrome.engine.offline": { tr: "Motor kapalı", en: "Engine offline" },
  "chrome.engine.ok": { tr: "Her şey yolunda.", en: "Everything is fine." },
  "chrome.engine.version": { tr: "Sürüm", en: "Version" },
  "chrome.engine.gpu": { tr: "GPU", en: "GPU" },
  "chrome.engine.gpu.none": { tr: "yok — CPU ile işlenecek", en: "none — will process on CPU" },
  "chrome.engine.disk": { tr: "Boş alan", en: "Free space" },
  "chrome.settings.soon": { tr: "Ayarlar yakında", en: "Settings coming soon" },

  // ── Property list ──────────────────────────────────────────────────────────────────────
  "properties.subtitle": { tr: "Oluşturduğunuz bütün evler.", en: "Every property you have added." },
  "properties.unit": { tr: "ev", en: "property" },
  "properties.unit.plural": { tr: "ev", en: "properties" },
  "properties.open": { tr: "Aç", en: "Open" },
  "properties.embed": { tr: "Embed kodu", en: "Embed code" },
  "properties.rename": { tr: "Yeniden adlandır", en: "Rename" },
  "properties.rename.title": { tr: "Evi yeniden adlandır", en: "Rename property" },
  "properties.delete": { tr: "Sil", en: "Delete" },
  "properties.delete.title": { tr: "Evi sil", en: "Delete property" },
  "properties.deleted": { tr: "Ev silindi.", en: "Property deleted." },
  "properties.renamed": { tr: "Ev adı güncellendi.", en: "Property renamed." },
  "properties.loadFailed": { tr: "Evler yüklenemedi", en: "Could not load properties" },
  "properties.noCover": { tr: "Kapak görseli yok", en: "No cover image" },

  // ── Generic ────────────────────────────────────────────────────────────────────────
  "common.back": { tr: "Geri", en: "Back" },
  "common.continue": { tr: "Devam", en: "Continue" },
  "common.close": { tr: "Kapat", en: "Close" },
  "common.copy": { tr: "Kopyala", en: "Copy" },
  "common.copyFailed": { tr: "Kopyalanamadı", en: "Copy failed" },
  "common.offline.title": { tr: "Motora ulaşılamıyor", en: "The engine is unreachable" },
  "common.loading": { tr: "Yükleniyor", en: "Loading" },
  "common.notFound": { tr: "Bulunamadı", en: "Not found" },
  "common.optional": { tr: "isteğe bağlı", en: "optional" },

  // ── Wizard ─────────────────────────────────────────────────────────────────────────
  "wizard.title": { tr: "Yeni ev", en: "New property" },
  "wizard.step.details": { tr: "Ev bilgileri", en: "Property details" },
  "wizard.step.capture": { tr: "Çekim", en: "Capture" },
  "wizard.step.upload": { tr: "Yükle", en: "Upload" },
  "wizard.step.details.desc": { tr: "İlanı tanıyacağınız bir ad verin.", en: "Give it a name you will recognise." },
  "wizard.step.capture.desc": { tr: "Doğru çekim, iyi bir turun yarısıdır.", en: "A good capture is half the tour." },
  "wizard.step.upload.desc": {
    tr: "Videoyu bu bilgisayardan bırakın ya da telefonla çekin.",
    en: "Drop the video from this computer, or shoot it with your phone.",
  },

  // The two ways footage gets in. Named by WHERE the person is standing, because that is the
  // actual choice: at a laptop with a file, or in the flat with a phone.
  "wizard.source.computer": { tr: "Bu bilgisayardan", en: "From this computer" },
  "wizard.source.phone": { tr: "Telefonla çek", en: "Shoot on the phone" },
  "wizard.phone.note": {
    tr: "Çekimi telefonda bitirdiğinizde oda bu eve kendiliğinden eklenir — bu sayfayı kapatabilirsiniz.",
    en: "When you finish on the phone the room is added to this property by itself — you can close this page.",
  },
  "wizard.phone.signin": {
    tr: "Telefonda ilk açılışta bir kez giriş yapmanız istenir.",
    en: "The phone asks you to sign in once, the first time.",
  },
  "wizard.phone.open": { tr: "Ev sayfasına git", en: "Go to the property" },
  "wizard.title.placeholder": { tr: "Örn. Bahçelievler 3+1 Daire", en: "e.g. Two-bedroom flat, Bahçelievler" },
  "wizard.address.placeholder": { tr: "Örn. Atatürk Cd. No:12, Kadıköy", en: "e.g. 12 Atatürk St, Kadıköy" },
  "wizard.title.required": { tr: "Ev başlığı gerekli.", en: "A property title is required." },
  "wizard.created": { tr: "Ev eklendi.", en: "Property added." },
  "wizard.scene.placeholder": { tr: "Örn. Salon", en: "e.g. Living room" },
  "wizard.quality": { tr: "İşleme kalitesi", en: "Processing quality" },
  "wizard.quality.desc": {
    tr: "Sonradan aynı videodan yeniden işleyebilirsiniz; seçim kalıcı değil.",
    en: "You can reprocess the same video later; this choice is not permanent.",
  },
  "wizard.quality.fast.trade": {
    tr: "En hızlı sonuç. İnce detaylarda hafif yumuşama olabilir.",
    en: "Fastest result. Fine detail may soften slightly.",
  },
  "wizard.quality.balanced.trade": {
    tr: "Çoğu ilan için doğru seçim: keskin 8K, makul süre.",
    en: "The right call for most listings: sharp 8K, sensible wait.",
  },
  "wizard.quality.studio.trade": {
    tr: "Mobilya kenarlarındaki paralaksı da onarır. En uzun süren mod.",
    en: "Also repairs parallax around furniture edges. The slowest mode.",
  },
  "capture.phone.label": { tr: "Telefonu kendi ekseninde çevirin", en: "Turn the phone around its own axis" },

  // ── Upload ─────────────────────────────────────────────────────────────────────────
  "upload.release": { tr: "Bırakın", en: "Drop it" },
  "upload.browse": { tr: "Dosya seç", en: "Choose file" },
  "upload.photosHint": {
    tr: "Fotoğraf serisi: 12–60 kare, aynı noktadan çekilmiş.",
    en: "Photo series: 12–60 frames shot from the same spot.",
  },
  "upload.reading": { tr: "Video okunuyor…", en: "Reading video…" },
  "upload.duration": { tr: "Süre", en: "Duration" },
  "upload.remove": { tr: "Kaldır", en: "Remove" },
  "upload.start": { tr: "Yükle ve işle", en: "Upload and process" },
  "upload.photosCount": { tr: "fotoğraf seçildi", en: "photos selected" },
  "upload.rejected": { tr: "Bu dosya türü desteklenmiyor.", en: "That file type is not supported." },
  "upload.tooBig": { tr: "Dosya 700 MB sınırını aşıyor.", en: "The file is over the 700 MB limit." },
  "upload.tooShort": {
    tr: "Video 5 saniyeden kısa. Motor tam bir tur göremeyebilir; yine de deneyebilirsiniz.",
    en: "The video is under 5 seconds. The engine may not see a full turn, but you can still try.",
  },
  "upload.tooLong": {
    tr: "Video 10 dakikadan uzun. İşleme süresi belirgin şekilde uzayacak.",
    en: "The video is over 10 minutes. Processing will take noticeably longer.",
  },
  "upload.fileTooLong": {
    tr: "Bir dosya 5 dakikadan uzun; motor dosya başına en fazla 5 dakika kabul eder ve bu yükleme reddedilir. Videoyu bölün ya da rehberli çekimi (Telefonla çek) kullanın.",
    en: "A file is longer than 5 minutes; the engine accepts at most 5 minutes per file and will refuse this upload. Split the video or use the guided capture (Shoot with a phone).",
  },
  "upload.durationUnknown": {
    tr: "Sürenin okunamadığı bir format — yine de yükleyebilirsiniz.",
    en: "The duration could not be read from this format — you can still upload it.",
  },

  // ── Job monitor ────────────────────────────────────────────────────────────────────
  "job.title": { tr: "Panorama hazırlanıyor", en: "Building the panorama" },
  // NOT "leave this page open". That was true of the UPLOAD, which dies with the tab, and
  // it is the opposite of true here: the job is queued on the server, its state is in the
  // database, and this view reattaches to it by id. Telling an agent to stand still and
  // watch would cost them the visit for no reason.
  "job.subtitle": {
    tr: "İşlem sunucuda sürüyor. Sayfayı kapatabilirsiniz; ilerlemeyi ev sayfasından takip edebilirsiniz.",
    en: "The engine is working on the server. You can close this page; the property keeps the progress.",
  },
  "job.steps": { tr: "İşlem adımları", en: "Pipeline stages" },
  "job.queued": { tr: "Sıraya alındı", en: "Queued" },
  "job.done.title": { tr: "Sahne hazır", en: "Scene is ready" },
  "job.done.body": { tr: "Düzenleyici açılıyor…", en: "Opening the editor…" },
  "job.openEditor": { tr: "Düzenleyiciye git", en: "Go to the editor" },
  "job.failed.title": { tr: "İşlem tamamlanamadı", en: "Processing did not finish" },
  "job.cancel.confirm": {
    tr: "İşlemi iptal ederseniz yüklenen video işlenmeden kalır. Devam edilsin mi?",
    en: "Cancelling leaves the uploaded video unprocessed. Continue?",
  },
  "job.cancelled": { tr: "İşlem iptal edildi.", en: "The job was cancelled." },
  "job.cancelFailed": { tr: "İşlem iptal edilemedi.", en: "The job could not be cancelled." },
  "job.retrying": { tr: "Yeniden işleniyor…", en: "Reprocessing…" },

  // ── Multi-room capture ─────────────────────────────────────────────────────────────
  // An agent shoots a whole flat in one visit and has no reason to stand in the hallway
  // watching a progress bar: the job is queued on the server and outlives the phone. These
  // strings exist to say that out loud and to put the next room one tap away.
  "job.nextRoom": { tr: "Sonraki odayı çek", en: "Capture the next room" },
  "job.nextRoom.body": {
    tr: "İşlem sunucuda sürüyor — telefonu kapatabilir, uygulamadan çıkabilirsiniz. Odaları çekip yolunuza devam edin; siz ofise varana kadar turlar hazır olur.",
    en: "Processing continues on the server — you can close the phone or leave the app. Shoot the rooms and carry on; the properties will be ready by the time you reach the office.",
  },
  "job.roomsCaptured": { tr: "Bu evde çekilen oda", en: "Rooms captured on this property" },
  "job.openProperty": { tr: "Evi aç", en: "Open the property" },

  // ── Property detail ────────────────────────────────────────────────────────────────────
  "detail.scenes": { tr: "Sahneler", en: "Scenes" },
  "detail.share": { tr: "Paylaşım", en: "Sharing" },
  "detail.publish.desc": {
    tr: "Yayındayken gömme bağlantısı bağlantıya sahip herkese açıktır.",
    en: "While published, anyone with the link can open the embed.",
  },
  "detail.published.on": { tr: "Ev yayına alındı.", en: "Property published." },
  "detail.published.off": { tr: "Ev yayından kaldırıldı.", en: "Property unpublished." },
  "detail.scene.edit": { tr: "Düzenle", en: "Edit" },
  "detail.scene.delete.confirm": {
    tr: "Bu sahne ve panoraması kalıcı olarak silinecek. Emin misiniz?",
    en: "This scene and its panorama will be permanently deleted. Are you sure?",
  },
  "detail.scene.delete.title": { tr: "Sahneyi sil", en: "Delete scene" },
  "detail.scene.deleted": { tr: "Sahne silindi.", en: "Scene deleted." },
  "detail.reprocess.started": { tr: "Yeniden işleme başladı.", en: "Reprocessing started." },
  "detail.empty.title": { tr: "Bu evde henüz sahne yok", en: "No scenes in this property yet" },
  "detail.empty.body": {
    tr: "İlk odanın videosunu yükleyerek başlayın. Her oda ayrı bir sahne olur.",
    en: "Start by uploading the first room's video. Each room becomes its own scene.",
  },
  "detail.notFound.title": { tr: "Ev bulunamadı", en: "Property not found" },
  "detail.notFound.body": {
    tr: "Bu ev silinmiş olabilir ya da bağlantı hatalı.",
    en: "It may have been deleted, or the link is wrong.",
  },
  "detail.addScene.hint": { tr: "Yeni bir oda yükleyin", en: "Upload another room" },
  "detail.updated": { tr: "Güncellendi", en: "Updated" },

  // ── Scene editor ───────────────────────────────────────────────────────────────────
  "editor.tab.tags": { tr: "Etiketler", en: "Tags" },
  "editor.tab.view": { tr: "Görünüm", en: "View" },
  "editor.tab.coverage": { tr: "Kapsama", en: "Coverage" },
  "editor.tab.report": { tr: "Rapor", en: "Report" },
  "editor.back": { tr: "Eve dön", en: "Back to property" },
  "editor.inspector": { tr: "Sahne denetçisi", en: "Scene inspector" },
  "editor.panel.hide": { tr: "Paneli gizle", en: "Hide panel" },
  "editor.panel.show": { tr: "Paneli göster", en: "Show panel" },
  "editor.hotspot.new": { tr: "Yeni etiket", en: "New tag" },
  "editor.hotspot.edit": { tr: "Etiketi düzenle", en: "Edit tag" },
  "editor.hotspot.type": { tr: "Tür", en: "Type" },
  "editor.hotspot.type.info": { tr: "Bilgi", en: "Info" },
  "editor.hotspot.type.link": { tr: "Odaya geçiş", en: "Room link" },
  "editor.hotspot.type.portal": { tr: "Zemin halkası", en: "Floor ring" },
  "editor.hotspot.type.media": { tr: "Medya", en: "Media" },
  "editor.hotspot.portal.hint": {
    tr: "Zemine tıklayın: halka, ışının zeminle kesiştiği yere oturur. Ufkun üstüne konursa çizilmez.",
    en: "Click the floor: the ring sits where the ray meets it. Placed above the horizon it is not drawn.",
  },
  "editor.hotspot.portal.above": {
    tr: "Bu nokta ufkun üstünde; halka zemine oturamaz. Aşağı bir yere taşıyın.",
    en: "This point is above the horizon, so the ring has no floor to sit on. Move it lower.",
  },
  "editor.hotspot.target": { tr: "Hedef oda", en: "Target room" },
  "editor.hotspot.target.placeholder": { tr: "Oda seçin", en: "Pick a room" },
  "editor.hotspot.media": { tr: "Medya bağlantısı", en: "Media URL" },
  "editor.hotspot.titleRequired": { tr: "Başlık gerekli.", en: "A title is required." },
  "editor.hotspot.targetRequired": { tr: "Bir hedef oda seçin.", en: "Pick a target room." },
  "editor.hotspot.saved": { tr: "Etiket kaydedildi.", en: "Tag saved." },
  "editor.hotspot.deleted": { tr: "Etiket silindi.", en: "Tag deleted." },
  "editor.hotspot.saveFailed": { tr: "Etiket kaydedilemedi.", en: "The tag could not be saved." },
  "editor.hotspot.deleteFailed": { tr: "Etiket silinemedi.", en: "The tag could not be deleted." },
  "editor.hotspot.empty": { tr: "Henüz etiket yok", en: "No tags yet" },
  "editor.hotspot.list": { tr: "Etiketler", en: "Tags" },
  "editor.hotspot.focus": { tr: "Bu etikete bak", en: "Look at this tag" },
  "editor.hotspot.noRooms": {
    tr: "Geçiş etiketi için bu evde başka hazır oda yok.",
    en: "There is no other ready room to link to.",
  },
  "editor.view.title": { tr: "Açılış görünümü", en: "Opening view" },
  "editor.view.desc": {
    tr: "Ziyaretçi turu açtığında kamera ilk bu yöne bakar. Panoramayı istediğiniz yöne çevirin ve kaydedin.",
    en: "The camera faces this way when a visitor opens the property. Turn the panorama and save.",
  },
  "editor.view.current": { tr: "Şu anki bakış", en: "Current view" },
  "editor.view.saved": { tr: "Açılış görünümü güncellendi.", en: "Opening view updated." },
  "editor.view.failed": { tr: "Açılış görünümü kaydedilemedi.", en: "The opening view could not be saved." },
  "editor.view.yaw": { tr: "Yatay", en: "Yaw" },
  "editor.view.pitch": { tr: "Dikey", en: "Pitch" },
  "editor.view.fov": { tr: "Görüş açısı", en: "Field of view" },
  "editor.notReady.title": { tr: "Sahne henüz hazır değil", en: "This scene is not ready yet" },
  "editor.notReady.body": {
    tr: "Panorama üretilmeden etiket ekleyemezsiniz. İşlem bitince bu sayfa kendini yeniler.",
    en: "You cannot tag a scene before its panorama exists. This page refreshes when processing ends.",
  },
  "editor.scene.notFound": { tr: "Sahne bulunamadı", en: "Scene not found" },

  // ── Coverage ───────────────────────────────────────────────────────────────────────
  "coverage.covered": { tr: "Kapsanan küre", en: "Sphere covered" },
  // Why a complete capture reads 89-91 and never 100. Three rows photograph a BAND; the two
  // polar caps are the engine's job, not the user's. Without this line the number looks like
  // a score the user is failing, and a correct capture gets re-shot for nothing.
  //
  // Phrased against the pass threshold rather than a ceiling on purpose. A real sweep wobbles
  // past the nominal lattice and can measure ABOVE the perfect-lattice maximum, so a sentence
  // naming a ceiling would be contradicted by the very number printed above it.
  "coverage.ceilingNote": {
    tr: "Bu normal: üç sıra küreyi bir kuşak olarak tarar, kalan tavan ve zemin tepelerini motor dolduruyor. %{target} üzeri tam çekim sayılır.",
    en: "This is expected: three rows photograph a band, and the engine fills the remaining ceiling and floor caps. Above {target}% counts as complete.",
  },
  "coverage.horizon": { tr: "Ufuk çizgisi", en: "Horizon" },
  "coverage.yawSpan": { tr: "Yatay tarama", en: "Horizontal sweep" },
  "coverage.pitchRange": { tr: "Dikey aralık", en: "Vertical range" },
  "coverage.zenith": { tr: "Tavan", en: "Ceiling" },
  "coverage.nadir": { tr: "Zemin", en: "Floor" },
  "coverage.yes": { tr: "kapsandı", en: "covered" },
  "coverage.no": { tr: "eksik", en: "missing" },
  "coverage.gapsTitle": { tr: "Eksik bölgeler", en: "Missing areas" },
  "coverage.map": { tr: "Kapsama haritası", en: "Coverage map" },
  "coverage.map.legend": {
    tr: "Kırmızı alanlar kameranın hiç görmediği yönlerdir.",
    en: "Red areas are directions the camera never saw.",
  },
  "coverage.none": { tr: "Bu sahne için kapsama analizi yok.", en: "No coverage analysis for this scene." },
  "dir.front": { tr: "önünüzde", en: "in front of you" },
  "dir.right": { tr: "sağınızda", en: "to your right" },
  "dir.back": { tr: "arkanızda", en: "behind you" },
  "dir.left": { tr: "solunuzda", en: "to your left" },
  "dir.up": { tr: "üst", en: "upper" },
  "dir.down": { tr: "alt", en: "lower" },
  "dir.level": { tr: "göz hizası", en: "eye level" },

  // ── Metrics ────────────────────────────────────────────────────────────────────────
  "metrics.none": { tr: "Bu sahne için teknik rapor yok.", en: "No technical report for this scene." },
  "metrics.frames": { tr: "Ayıklanan kare", en: "Frames extracted" },
  "metrics.pairs": { tr: "Eşleşen kare çifti", en: "Matched pairs" },
  "metrics.loops": { tr: "Kapanan döngü", en: "Loop closures" },
  "metrics.exposure": { tr: "Pozlama farkı", en: "Exposure spread" },
  "metrics.pattern": { tr: "Tarama deseni", en: "Sweep pattern" },
  "metrics.pattern.single_row": { tr: "Tek sıra", en: "Single row" },
  "metrics.pattern.multi_row": { tr: "Çok sıralı", en: "Multi row" },
  "metrics.pattern.unknown": { tr: "Belirsiz", en: "Unknown" },
  "metrics.timing": { tr: "Süre dağılımı", en: "Time per stage" },
  "metrics.device": { tr: "Cihaz ve objektif", en: "Device and lens" },
  "metrics.warnings": { tr: "Motor uyarıları", en: "Engine warnings" },
  "metrics.verdict.great": { tr: "Mükemmel", en: "Excellent" },
  "metrics.verdict.ok": { tr: "İyi", en: "Good" },
  "metrics.verdict.poor": { tr: "Zayıf", en: "Weak" },
  "metrics.explain.keyframes": {
    tr: "Bir tam tur için 24–60 kare idealdir; çok azı boşluk, çok fazlası gereksiz süre demek.",
    en: "24–60 frames per full turn is ideal; too few leaves gaps, too many just costs time.",
  },
  "metrics.explain.reprojection": {
    tr: "Karelerin ne kadar üst üste oturduğu. 1,5 pikselin altı mükemmel, 3 pikselin üstünde dikişler görünür.",
    en: "How precisely frames line up. Under 1.5 px is excellent; over 3 px the seams start to show.",
  },
  "metrics.explain.loop": {
    tr: "Turun başladığı yere dönüş hatası. 1°’nin altı mükemmel; büyük değerler ufkun kaymasına yol açar.",
    en: "Error when the sweep returns to its start. Under 1° is excellent; large values tilt the horizon.",
  },
  "metrics.explain.hfov": {
    tr: "Telefon lensinin yatay görüş açısı. Tipik bir arka kamera 60°–75° arasındadır.",
    en: "The lens's horizontal field of view. A typical rear camera sits between 60° and 75°.",
  },
  "metrics.explain.seam": {
    tr: "Birleşim yerlerindeki parlaklık sıçraması. Düşük değer, dikişin göze çarpmadığı anlamına gelir.",
    en: "Brightness jump across the seams. Lower means the seam is invisible.",
  },
  "metrics.explain.exposure": {
    tr: "Kareler arasındaki pozlama farkı. Düşük değer, gökyüzü ve duvarlarda leke oluşmadığını gösterir.",
    en: "Exposure difference between frames. Low values mean no blotches on skies and walls.",
  },
  "metrics.explain.pairs": {
    tr: "Birbirine bağlanabilen kare çifti sayısı. Ne kadar çoksa çözüm o kadar kararlı.",
    en: "How many frame pairs could be linked. More pairs, more stable solve.",
  },

  // ── Embed ──────────────────────────────────────────────────────────────────────────
  "embed.responsive": { tr: "Duyarlı sarmalayıcı", en: "Responsive wrapper" },
  "embed.responsive.desc": {
    tr: "Sabit en-boy oranı isteyen portallar için 16:9 sarmalayıcı ekler.",
    en: "Adds a 16:9 wrapper for portals that need a fixed aspect ratio.",
  },
  "embed.open": { tr: "Turu yeni sekmede aç", en: "Open the tour in a new tab" },
  "embed.link": { tr: "Doğrudan bağlantı", en: "Direct link" },
  // ── Oda fotoğrafları ───────────────────────────────────────────────────────────────
  "photos.tab": { tr: "Fotoğraflar", en: "Photos" },
  "photos.drop": { tr: "Fotoğrafları buraya sürükleyin", en: "Drop photos here" },
  "photos.hint": { tr: "JPEG / PNG / WebP · bir seferde en fazla 24 dosya", en: "JPEG / PNG / WebP · up to 24 files at once" },
  "photos.choose": { tr: "veya dosya seçin", en: "or choose files" },
  "photos.empty": { tr: "Bu odada henüz fotoğraf yok", en: "No photos in this room yet" },
  "photos.uploading": { tr: "Yükleniyor…", en: "Uploading…" },
  "photos.delete": { tr: "Fotoğrafı sil", en: "Delete photo" },
  "photos.caption": { tr: "Açıklama", en: "Caption" },
  "photos.count": { tr: "fotoğraf", en: "photos" },
  "photos.skipped": {
    tr: "{n} dosya okunamadı ve atlandı.",
    en: "{n} file(s) could not be read and were skipped.",
  },
  "photos.failed": { tr: "Fotoğraflar yüklenemedi.", en: "The photos could not be uploaded." },
  "photos.deleted": { tr: "Fotoğraf silindi.", en: "Photo deleted." },
  "beta.hint": {
    tr: "Bu özellik beta aşamasında; sonuçlar her karede aynı kalitede olmayabilir.",
    en: "This feature is in beta; results are not equally good on every frame.",
  },
  "photos.studio.open": { tr: "Düzenle", en: "Edit" },
  "photos.studio.title": { tr: "Fotoğraf düzenle", en: "Edit photo" },
  "photos.studio.styles": { tr: "Dekorasyon", en: "Staging" },
  "photos.studio.empty": { tr: "Mobilyaları kaldır", en: "Remove furniture" },
  "photos.studio.original": { tr: "Orijinal", en: "Original" },
  "photos.studio.working": { tr: "İşleniyor…", en: "Working..." },
  "photos.studio.nothing": {
    tr: "Bu karede kaldırılacak bir mobilya bulunamadı.",
    en: "No furnishings were found in this frame.",
  },
  "photos.studio.removed": {
    tr: "Karenin yaklaşık %{n} kadarı yeniden oluşturuldu.",
    en: "{n}% of the frame was reconstructed.",
  },
  "photos.studio.kept": {
    tr: "Karenin yaklaşık %{n} kadarı orijinalden korundu.",
    en: "{n}% of the frame was kept from the original.",
  },
  "photos.studio.found": { tr: "Odada bulunanlar", en: "Found in the room" },
  "photos.studio.soft": {
    tr: "Dolgu yumuşak kalabilir; mobilyanın arkası hiç fotoğraflanmadığı için yeniden oluşturuluyor.",
    en: "The fill can be soft: what was behind the furniture was never photographed, so it is reconstructed.",
  },
  "room.add": { tr: "Oda ekle", en: "Add room" },

  "walk.add": { tr: "3B Gezinti", en: "3D Walkthrough" },
  "walk.add.title": { tr: "3B gezinti oluştur", en: "Build a 3D walkthrough" },
  "walk.add.desc": {
    tr: "Bir genel gezinti videosu ve her oda için bir video yükleyin. Sistem kareleri hizalar, geometriyi çözer ve sahneyi eğitir; ziyaretçi dairede serbestçe yürür.",
    en: "Upload one general walkthrough video and one video per room. The system aligns the frames, solves the geometry and trains the scene; the visitor walks the flat freely.",
  },
  "walk.base.title": { tr: "Genel gezinti", en: "General walkthrough" },
  "walk.base.hint": {
    tr: "Tek video, kapıdan başlayıp bütün daireyi yavaşça dolaşan 1–3 dakika. Odaları birbirine bağlar; birden fazla oda varsa zorunlu.",
    en: "One 1–3 minute video walking slowly through the whole flat from the front door. It connects the rooms; required when there is more than one.",
  },
  "walk.base.pick": { tr: "Genel gezinti videosu seç", en: "Choose the walkthrough video" },
  "walk.rooms.title": { tr: "Odalar", en: "Rooms" },
  "walk.rooms.hint": {
    tr: "Oda başına 2–4 dakikalık bir video: kapı eşiğinden başlayıp çevreyi iki yönde dolaşın, çapraz geçin, detaylara yaklaşın.",
    en: "One 2–4 minute video per room: start under the door frame, loop the perimeter both ways, cross the room, get close to details.",
  },
  "walk.rooms.add": { tr: "Oda ekle", en: "Add room" },
  "walk.room.placeholder": { tr: "Oda {n}", en: "Room {n}" },
  "walk.room.new": { tr: "Yeni oda", en: "New room" },
  "walk.room.existing": { tr: "Mevcut oda", en: "Existing room" },
  "walk.room.pick": { tr: "Oda videosu seç", en: "Choose the room video" },
  "walk.needsBase": {
    tr: "Birden fazla oda için genel gezinti videosu gerekli: odaları daireye yerleştiren odur.",
    en: "More than one room needs the general walkthrough video: it is what places the rooms in the flat.",
  },
  "walk.duplicateNames": { tr: "Aynı adlı dosyalar var; her videonun adı farklı olmalı:", en: "Duplicate file names; every video needs a distinct name:" },
  "walk.start": { tr: "Yükle ve oluştur", en: "Upload and build" },
  "walk.open": { tr: "3B gezintiyi aç", en: "Open the 3D walkthrough" },
  "walk.rebuild": { tr: "Yeniden oluştur", en: "Rebuild" },
  "walk.capturePhone": { tr: "Telefonla çek", en: "Shoot with a phone" },
  "walk.card.warnings": { tr: "Çekim uyarıları", en: "Capture warnings" },
  "walk.job.title": { tr: "3B gezinti hazırlanıyor", en: "Building the 3D walkthrough" },
  "walk.quality.fast.desc": { tr: "12K adım · SH1 · ~5 dk/alan", en: "12K steps · SH1 · ~5 min/area" },
  "walk.quality.balanced.desc": { tr: "30K adım · SH2 · ~15 dk/alan", en: "30K steps · SH2 · ~15 min/area" },
  "walk.quality.studio.desc": { tr: "40K adım · SH3 · ~25 dk/alan", en: "40K steps · SH3 · ~25 min/area" },
  "walk.quality.fast.trade": {
    tr: "Hızlı bir önizleme: ince dokular yumuşar, uzak köşelerde sis kalabilir.",
    en: "A quick preview: fine texture softens and far corners can stay hazy.",
  },
  "walk.quality.balanced.trade": {
    tr: "Çoğu ilan için doğru seçim: keskin oda, makul süre, küçük dosya.",
    en: "The right call for most listings: a sharp room, sensible wait, small file.",
  },
  "walk.quality.studio.trade": {
    tr: "En keskin detay ve en uzun süre; büyük ya da eşyası çok odalar için.",
    en: "Sharpest detail and the longest wait; for large or busy rooms.",
  },
  "walk.capturePhone.short": { tr: "Telefonla çek", en: "Phone" },
  "walk.remove": { tr: "3B gezintiyi kaldır", en: "Remove the 3D walkthrough" },
  "walk.remove.confirm": {
    tr: "3B gezinti, videoları ve odaların 3B alanları silinecek. Odaların 360 panoramaları kalır. Emin misiniz?",
    en: "The 3D walkthrough, its videos and the rooms' 3D areas will be deleted. The rooms' 360 panoramas stay. Are you sure?",
  },
  "walk.card.title": { tr: "3B Gezinti", en: "3D Walkthrough" },
  "walk.card.none": {
    tr: "Henüz 3B gezinti yok. Bir genel gezinti ve oda videoları yükleyin; ziyaretçi dairede serbestçe yürüsün.",
    en: "No 3D walkthrough yet. Upload a general walkthrough and room videos; let the visitor walk the flat freely.",
  },
  "walk.card.processing": { tr: "3B sahne oluşturuluyor. Bu bir saatten uzun sürebilir; sekmeyi kapatabilirsiniz.", en: "Building the 3D scene. This can take over an hour; you can close the tab." },
  "walk.card.failed": { tr: "Oluşturma başarısız oldu.", en: "The build failed." },
  "walk.card.ready": { tr: "Hazır", en: "Ready" },
  "walk.empty.title": { tr: "3B gezinti yok", en: "No 3D walkthrough" },
  "walk.empty.body": { tr: "Ev sayfasından bir genel gezinti ve oda videoları yükleyin.", en: "Upload a general walkthrough and room videos from the property page." },
  "walk.failed.title": { tr: "3B gezinti oluşturulamadı", en: "The 3D walkthrough could not be built" },
  "walk.stat.areas": { tr: "Alan", en: "Areas" },
  "walk.stat.splats": { tr: "Splat", en: "Splats" },
  "walk.stat.size": { tr: "Boyut", en: "Size" },
  "walk.stat.time": { tr: "Süre", en: "Build time" },
  "walk.tile.open": { tr: "3B alanı aç", en: "Open the 3D area" },

  "walkcap.title": { tr: "3B gezinti çekimi", en: "3D walkthrough capture" },
  "walkcap.plan.intro": {
    tr: "Önce bütün daireyi dolaşan bir genel gezinti, sonra her oda için ayrı bir video. Her alanı çektikten sonra hemen yüklenir; hepsi bittiğinde 3B sahne oluşturulur.",
    en: "First one general walkthrough through the whole flat, then one video per room. Each area uploads right after it is shot; the 3D scene is built when all are in.",
  },
  "walkcap.plan.general": { tr: "Genel gezinti", en: "General walkthrough" },
  "walkcap.plan.general.hint": { tr: "1–3 dk · kapıdan başla, her odadan 8–10 sn geç", en: "1–3 min · start at the door, 8–10 s through every room" },
  "walkcap.plan.rooms": { tr: "Odalar", en: "Rooms" },
  "walkcap.plan.room.hint": { tr: "2–4 dk · kapı eşiğinden başla, 4 geçiş", en: "2–4 min · start under the door frame, 4 passes" },
  "walkcap.plan.addRoom": { tr: "Oda ekle", en: "Add room" },
  "walkcap.plan.start": { tr: "Çekime başla", en: "Start shooting" },
  "walkcap.plan.skipGeneral": { tr: "Tek oda: genel gezinti gerekmez", en: "One room: no general walkthrough needed" },
  "walkcap.setup.title": { tr: "Kamera hazırlanıyor", en: "Preparing the camera" },
  "walkcap.setup.lens": { tr: "Ultra geniş lens", en: "Ultra-wide lens" },
  "walkcap.setup.lens.fallback": { tr: "Ultra geniş lens bulunamadı; geniş lensle devam", en: "No ultra-wide lens found; continuing with the wide one" },
  "walkcap.setup.portrait": { tr: "Telefonu dik tutun", en: "Hold the phone upright" },
  "walkcap.rules.1": { tr: "Telefon dik, göğüs–baş hizasında, dümdüz ileri.", en: "Phone upright, chest-to-head height, straight ahead." },
  "walkcap.rules.2": { tr: "Çok yavaş yürüyün: saniyede bir adım.", en: "Walk very slowly: one step per second." },
  "walkcap.rules.3": { tr: "Dönüşlerde durun, yavaşça dönün, sonra devam edin.", en: "To turn: stop, turn slowly, continue." },
  "walkcap.rules.4": { tr: "Boş duvara ya da yalnız zemine bakmayın; detay görünsün.", en: "Do not point at bare walls or only the floor; keep detail in view." },
  "walkcap.record": { tr: "Kaydı başlat", en: "Start recording" },
  "walkcap.stop": { tr: "Bitir", en: "Finish" },
  "walkcap.nextPass": { tr: "Sonraki geçiş", en: "Next pass" },
  "walkcap.pass.1": { tr: "1. geçiş · Çevre, bir yön", en: "Pass 1 · Perimeter, one way" },
  "walkcap.pass.1.cue": { tr: "Duvarlardan ~1.5 m uzaktan odanın çevresini dolaşın.", en: "Walk the perimeter about 1.5 m from the walls." },
  "walkcap.pass.2": { tr: "2. geçiş · Çevre, ters yön", en: "Pass 2 · Perimeter, the other way" },
  "walkcap.pass.2.cue": { tr: "Aynı turu ters yönde yapın; her duvar iki yönden görünsün.", en: "Same loop the other way; every wall seen from both directions." },
  "walkcap.pass.3": { tr: "3. geçiş · Çapraz", en: "Pass 3 · Cross" },
  "walkcap.pass.3.cue": { tr: "Girişten karşı köşeye yürüyün, yavaşça dönün, geri gelin.", en: "Walk from the entrance to the far corner, turn slowly, come back." },
  "walkcap.pass.4": { tr: "4. geçiş · Detaylar", en: "Pass 4 · Details" },
  "walkcap.pass.4.cue": { tr: "Dokulu yüzeylere 30–60 cm yaklaşın: tezgâh, raf, tablo.", en: "Get within 30–60 cm of textured surfaces: counter, shelves, pictures." },
  "walkcap.general.cue": { tr: "Kapıdan başlayın; her odaya girin, ortasından 8–10 saniyede geçin, çıkın. Durmayın.", en: "Start at the door; enter every room, cross its middle in 8–10 s, leave. Keep moving." },
  "walkcap.tooFast": { tr: "Çok hızlı dönüyorsunuz", en: "Turning too fast" },
  "walkcap.tooFast.body": { tr: "Durun, sonra yavaşça dönün; hızlı dönüş kareleri bulanıklaştırır.", en: "Stop, then turn slowly; a fast pan blurs the frames." },
  "walkcap.landscape": { tr: "Telefonu dik tutun", en: "Hold the phone upright" },
  "walkcap.review.title": { tr: "Kayıt tamam", en: "Recording done" },
  "walkcap.review.retake": { tr: "Yeniden çek", en: "Retake" },
  "walkcap.review.upload": { tr: "Yükle ve devam et", en: "Upload and continue" },
  "walkcap.review.short": { tr: "Kayıt çok kısa; en az 60 saniye çekin.", en: "The recording is too short; shoot at least 60 seconds." },
  "walkcap.uploading": { tr: "Yükleniyor", en: "Uploading" },
  "walkcap.uploaded": { tr: "Yüklendi", en: "Uploaded" },
  "walkcap.next.title": { tr: "Sıradaki alan", en: "Next area" },
  "walkcap.build": { tr: "3B sahneyi oluştur", en: "Build the 3D scene" },
  "walkcap.build.hint": { tr: "Bütün alanlar yüklendi. Oluşturma bir saatten uzun sürebilir; telefonu kapatabilirsiniz.", en: "Every area is in. Building can take over an hour; you can put the phone away." },
  "walkcap.done.open": { tr: "3B gezintiyi aç", en: "Open the 3D walkthrough" },
  "walkcap.elapsed": { tr: "süre", en: "elapsed" },
  "walkcap.cameraError": { tr: "Kamera açılamadı", en: "The camera could not be opened" },
  "room.add.title": { tr: "Fotoğraflı oda ekle", en: "Add a photo-only room" },
  "room.add.body": {
    tr: "360 çekimi olmayan bir oda oluşturun; içine fotoğraf ekleyebilirsiniz. 360'ı sonradan da çekebilirsiniz.",
    en: "Create a room with no 360 capture; you can put photos in it. The 360 can come later.",
  },
  "room.add.name": { tr: "Oda adı", en: "Room name" },
  "room.add.placeholder": { tr: "Çocuk odası", en: "Child's room" },
  "room.added": { tr: "Oda eklendi.", en: "Room added." },
  "room.nocapture": { tr: "360 çekimi yok", en: "No 360 capture" },
  "room.photoOnly.title": { tr: "Fotoğraflı oda", en: "Photo-only room" },
  "room.photoOnly.body": {
    tr: "Bu odanın 360 çekimi yok. Fotoğraf ekleyebilir, dilediğinizde 360 da çekebilirsiniz.",
    en: "This room has no 360 capture. Add photos, and shoot the 360 whenever you like.",
  },

  // ── Sanal dekorasyon ───────────────────────────────────────────────────────────────
  "decor.title": { tr: "Sanal Dekorasyon", en: "Virtual Staging" },
  "decor.subtitle": {
    tr: "Bir oda fotoğrafı yükleyin; aynı odayı beş farklı dekorasyon stilinde görün.",
    en: "Upload a room photo and see the same room in five decorating styles.",
  },
  "decor.nav": { tr: "Dekorasyon", en: "Staging" },
  "decor.drop": { tr: "Oda fotoğrafını buraya sürükleyin", en: "Drop a room photo here" },
  "decor.drop.hint": { tr: "JPEG / PNG / WebP · en fazla 32 MB", en: "JPEG / PNG / WebP · up to 32 MB" },
  "decor.choose": { tr: "veya dosya seçin", en: "or choose a file" },
  "decor.styles": { tr: "Dekorasyon stilleri", en: "Decorating styles" },
  "decor.original": { tr: "Orijinal fotoğraf", en: "Original photo" },
  "decor.generated": { tr: "Yapay zeka", en: "AI generated" },
  "decor.slider": { tr: "Öncesi ve sonrası karşılaştırma çubuğu", en: "Before and after comparison slider" },
  "decor.working": { tr: "Oda döşeniyor…", en: "Furnishing the room…" },
  "decor.queued": { tr: "Sırada", en: "Queued" },
  "decor.uploading": { tr: "Fotoğraf yükleniyor…", en: "Uploading the photo…" },
  "decor.retry": { tr: "Tekrar dene", en: "Try again" },
  "decor.variation": { tr: "Başka bir öneri", en: "Another take" },
  "decor.variation.hint": {
    tr: "Aynı stilin farklı bir yorumunu üretir; önceki sonuç bununla değişir.",
    en: "Renders a different take of the same style; it replaces the one on screen.",
  },
  "decor.another": { tr: "Başka fotoğraf", en: "Another photo" },
  "decor.download": { tr: "Görseli indir", en: "Download image" },
  "decor.failed": { tr: "Bu stil oluşturulamadı.", en: "This style could not be generated." },
  "decor.unavailable.title": { tr: "Sanal dekorasyon bu sunucuda kapalı", en: "Virtual staging is off on this server" },
  "decor.unavailable.body": {
    tr: "Özellik CUDA destekli bir ekran kartı ve difüzyon ağırlıkları ister. Motoru o makinede çalıştırın ya da BIOSPHERE_AI_WORKER_URL ile iyi kartı olan makineye yönlendirin.",
    en: "The feature needs a CUDA GPU with the diffusion weights present. Run the engine on that machine, or point BIOSPHERE_AI_WORKER_URL at the box with the good card.",
  },
  "decor.protected": {
    tr: "Karenin yaklaşık %{n} kadarı orijinalden korundu: pencereler, kapılar, tesisat ve dışarıda görünen manzara.",
    en: "{n}% of the frame was kept from the original: windows, doors, fittings and the view outside.",
  },
  "decor.disclaimer": {
    tr: "Mobilya ve yüzeyler yapay zeka ile üretildi; odanın mimarisi korunur. Bu bir dekorasyon önerisidir, mülkün mevcut hali değildir.",
    en: "Furniture and finishes are AI generated; the room's architecture is preserved. This is a decorating proposal, not the property as it stands.",
  },
  "decor.notimage": { tr: "Bu dosya bir görsel değil.", en: "That file is not an image." },

  "embed.draft": {
    tr: "Ev taslak durumda. Paylaşmadan önce yayına almayı unutmayın.",
    en: "This property is a draft. Remember to publish before you share it.",
  },

  // ── İlan metni ─────────────────────────────────────────────────────────────────────
  "listing.title": { tr: "İlan başlığı ve açıklaması", en: "Listing headline and description" },
  "listing.body": {
    tr: "sahibinden.com'a yapıştırılacak biçimde, evin kendisinden yazılır: oda adları, ölçülen alanlar, tavan yüksekliği ve odaların kendi görüntüleri. Ölçemediklerini (kat, cephe, ısıtma, 3+1) ek nota yazın. Metin size ait — düzenleyip kaydedin.",
    en: "Written for sahibinden.com from the property itself: room names, measured areas, ceiling height and the rooms' own views. Put what it cannot measure — floor, aspect, heating, layout — in the extra instruction. The text is yours: edit it, then save.",
  },
  "listing.images": { tr: "oda görüntüsü", en: "room views" },
  "listing.offline": { tr: "Modelsiz yazıldı", en: "Composed without a model" },
  "listing.cost": { tr: "Maliyet", en: "Cost" },
  "listing.cost.cached": { tr: "Önbellek · ilk yazımı", en: "Cached · first write cost" },
  "listing.cost.unknown": { tr: "Maliyet bilinmiyor", en: "Cost unknown" },
  "listing.seconds": { tr: "sn", en: "s" },
  "listing.tone": { tr: "Üslup", en: "Tone" },
  "listing.tone.professional": { tr: "Kurumsal", en: "Professional" },
  "listing.tone.warm": { tr: "Sıcak", en: "Warm" },
  "listing.tone.concise": { tr: "Kısa ve net", en: "Concise" },
  "listing.tone.premium": { tr: "Üst segment", en: "Premium" },
  "listing.instructions": { tr: "Ek not (isteğe bağlı)", en: "Extra instruction (optional)" },
  "listing.instructions.placeholder": {
    tr: "ör. 3+1, 8. kat, güney cephe, asansörlü — bunlar ölçülemez, siz yazın",
    en: "e.g. 3+1, 8th floor, south facing, lift — it cannot measure these, you tell it",
  },
  "listing.write": { tr: "Metni yaz", en: "Write the copy" },
  "listing.rewrite": { tr: "Yeniden yaz", en: "Write again" },
  "listing.again": { tr: "Başka bir metin dene", en: "Try a different one" },
  "listing.warnings": {
    tr: "Bu metinde evden doğrulanamayan ifadeler var",
    en: "This text makes claims the property cannot support",
  },
  "listing.field.title": { tr: "Başlık", en: "Headline" },
  "listing.field.description": { tr: "Açıklama", en: "Description" },
  "listing.alts": { tr: "Diğer başlıklar:", en: "Other headlines:" },
  "listing.words": { tr: "kelime", en: "words" },
  "listing.highlights": { tr: "Öne çıkanlar", en: "Highlights" },
  "listing.save": { tr: "Eve kaydet", en: "Save to the property" },
  "listing.saved": { tr: "İlan metni kaydedildi.", en: "The listing copy was saved." },
  "listing.copy": { tr: "Kopyala", en: "Copy" },
  "listing.copied": { tr: "Kopyalandı", en: "Copied" },
  "listing.upToDate": { tr: "Evdeki metinle aynı.", en: "Same as the property's own copy." },
  "listing.failed": { tr: "İlan metni yazılamadı.", en: "The listing copy could not be written." },
  "listing.facts": { tr: "Bu metin şunlardan yazıldı", en: "Written from" },
  "listing.facts.rooms": { tr: "Oda", en: "Rooms" },
  "listing.facts.measured": { tr: "Ölçülen oda", en: "Measured" },
  "listing.facts.area": { tr: "Ölçülen alan", en: "Measured area" },
  "listing.facts.ceiling": { tr: "Tavan", en: "Ceiling" },
  "listing.facts.note": {
    tr: "Alan yalnızca ölçülen odaların toplamıdır; koridorlar ve ölçülmemiş odalar dahil değildir.",
    en: "The area is the sum of the measured rooms only; hallways and unmeasured rooms are not in it.",
  },
  "listing.facts.unmeasured": {
    tr: "Hiçbir oda ölçülmemiş, bu yüzden metinde alan geçmiyor. Aşağıdaki kat planından odaları ölçerseniz metin daha somut olur.",
    en: "No room has been measured, so the copy states no area. Measure the rooms on the plan below and the copy gets concrete.",
  },
} satisfies Dict;

export type UiKey = keyof typeof uiStrings;

export function ui(key: UiKey, locale: Locale = "tr"): string {
  return uiStrings[key][locale];
}

/**
 * Turkish for what the scene parser found standing in the room.
 *
 * The engine reports ADE20K's own class names because they are the segmenter's vocabulary and
 * the API is not the place to guess a display language — but "bed, chair, stool, table" under
 * a Turkish sentence is the API leaking into the product. Anything not listed falls through
 * unchanged, which is the right failure: an English noun the user can still read beats a blank
 * where a noun should be.
 */
const OBJECT_TR: Record<string, string> = {
  bed: "yatak",
  chair: "sandalye",
  "swivel chair": "çalışma sandalyesi",
  "office chair": "çalışma sandalyesi",
  armchair: "koltuk",
  sofa: "kanepe",
  bench: "bank",
  stool: "tabure",
  ottoman: "puf",
  table: "masa",
  "coffee table": "sehpa",
  desk: "çalışma masası",
  cabinet: "dolap",
  wardrobe: "gardırop",
  "chest of drawers": "şifonyer",
  sideboard: "konsol",
  "display cabinet": "vitrin",
  bookcase: "kitaplık",
  shelf: "raf",
  counter: "tezgâh",
  lamp: "lamba",
  light: "aydınlatma",
  chandelier: "avize",
  sconce: "aplik",
  rug: "halı",
  carpet: "halı",
  cushion: "minder",
  pillow: "yastık",
  blanket: "battaniye",
  curtain: "perde",
  mirror: "ayna",
  painting: "tablo",
  poster: "poster",
  "bulletin board": "pano",
  screen: "paravan",
  television: "televizyon",
  monitor: "monitör",
  computer: "bilgisayar",
  fan: "vantilatör",
  radiator: "radyatör",
  box: "kutu",
  basket: "sepet",
  bag: "çanta",
  vase: "vazo",
  pot: "saksı",
  flower: "çiçek",
  book: "kitap",
  clothes: "kıyafet",
  toy: "oyuncak",
  bin: "çöp kovası",
  clock: "saat",
  bottle: "şişe",
  tray: "tepsi",
  plate: "tabak",
  towel: "havlu",
  sculpture: "biblo",
  cradle: "beşik",
  stove: "ocak",
  refrigerator: "buzdolabı",
};

export function roomObject(name: string, locale: Locale = "tr"): string {
  if (locale !== "tr") return name;
  return OBJECT_TR[name.trim().toLowerCase()] ?? name;
}
