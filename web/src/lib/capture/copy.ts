/**
 * Guided-capture copy — Turkish first, English second.
 *
 * `lib/i18n.ts` is frozen, so every string the capture flow needs lives here, in exactly
 * the shape of `components/dashboard/copy.tsx`. The `satisfies Record<CaptureCopyKey, …>`
 * at the bottom is load-bearing: a key added to the contract's union but not given a value
 * here is a COMPILE error, not a "??key??" appearing on a phone screen mid-sweep, where
 * the user is turning on the spot and cannot stop to read anything anyway.
 *
 * TONE RULES, because this copy is read at arm's length while the reader is physically
 * rotating:
 *  - Warnings are one or two words. The user gets a ~200 ms glance, not a sentence.
 *  - Bodies are one short clause and always name the FIX, never the fault: "Ayaklarınız
 *    sabit kalsın", not "Yürüyorsunuz".
 *  - Formal plural ("çevirin", "tutun") throughout, matching the rest of the product.
 *    It costs one syllable over the singular and keeps one voice across the app.
 *  - Turkish is written as Turkish, not as a translation of the English: "Olduğunuz yerde
 *    dönün" is what a person actually says, "kendi ekseninizde dönün" is what a manual
 *    says.
 *
 * INTERPOLATION: `{n}` and `{band}` are substituted by the caller via `cc(key, locale,
 * vars)`. Both placeholders must survive in BOTH languages — a missing `{n}` in one
 * language silently drops the number for half the users.
 */

import type { Locale } from "@/lib/types";
import type { CaptureCopyKey } from "./types";

/** Same shape as the dashboard's local dictionary and as `lib/i18n.ts`. */
type Dict = Record<CaptureCopyKey, { tr: string; en: string }>;

export const captureStrings = {
  // ── Intro ────────────────────────────────────────────────────────────────────────────
  "capture.intro.title": { tr: "Telefonla 360° çekim", en: "Shoot 360° with your phone" },
  "capture.intro.body": {
    tr: "Odanın ortasında durun, telefonu kendi ekseninde üç tur çevirin. Yaklaşık bir dakika sürer; hizalamayı motor yapar.",
    en: "Stand in the middle of the room and turn the phone around its own axis three times. It takes about a minute; the engine does the aligning.",
  },
  "capture.intro.start": { tr: "Başla", en: "Start" },
  "capture.intro.have": { tr: "Hazır videom var", en: "I already have a video" },
  "capture.intro.how": { tr: "Nasıl çekilir?", en: "How does it work?" },

  // ── Permissions ──────────────────────────────────────────────────────────────────────
  "capture.perm.title": { tr: "İki izin gerekiyor", en: "Two permissions are needed" },
  "capture.perm.body": {
    tr: "Kamera görüntüyü kaydeder. Hareket sensörü telefonun nereye baktığını söyler — rehber halkalar onunla çalışır. İkisi de bu sayfadan çıkmaz.",
    en: "The camera records the picture. The motion sensor tells us where the phone is pointing — that is what the guide rings run on. Neither leaves this page.",
  },
  "capture.perm.allow": { tr: "İzin ver", en: "Allow" },
  "capture.perm.camera.denied": {
    tr: "Kamera izni verilmedi. Adres çubuğundaki kilit simgesinden bu siteye kamera iznini açın, sonra tekrar deneyin.",
    en: "Camera access was refused. Open the lock icon in the address bar, allow the camera for this site, then try again.",
  },
  "capture.perm.motion.denied": {
    tr: "Hareket sensörü kapalı. Çekim, süreli basit rehberle devam eder — halkalar yerine sayaç görürsünüz. Açmak için: Ayarlar › Safari › Hareket ve Yön Erişimi.",
    en: "The motion sensor is off. The capture continues with a simple timed guide — a counter instead of rings. To turn it on: Settings › Safari › Motion & Orientation Access.",
  },
  "capture.perm.inapp": { tr: "Tarayıcıda açın", en: "Open in your browser" },
  "capture.perm.inapp.body": {
    tr: "Instagram, Facebook ve benzeri uygulama içi tarayıcılar kameraya erişemiyor. Bu bağlantıyı Safari veya Chrome'da açın.",
    en: "In-app browsers like Instagram's and Facebook's cannot reach the camera. Open this link in Safari or Chrome.",
  },

  // ── Orientation interrupt ────────────────────────────────────────────────────────────
  "capture.rotate.title": { tr: "Dikey tutun", en: "Hold it upright" },
  "capture.rotate.body": {
    tr: "Kayıt bekliyor. Telefonu dikeye çevirince kaldığı yerden devam eder — hiçbir şey kaybolmaz.",
    en: "Recording is waiting. Turn the phone upright and it picks up where it left off — nothing is lost.",
  },

  // ── Stance ───────────────────────────────────────────────────────────────────────────
  "capture.stand.title": { tr: "Çekimden önce", en: "Before you start" },
  "capture.stand.1": {
    tr: "Odanın ortasında durun. Ayaklarınız çekim boyunca yerinden oynamasın.",
    en: "Stand in the middle of the room. Your feet stay where they are for the whole capture.",
  },
  "capture.stand.2": {
    tr: "Telefonu iki elle, dirsekler gövdenize bitişik, göz hizasında ve dikey tutun.",
    en: "Hold the phone with both hands, elbows against your body, at eye level and upright.",
  },
  "capture.stand.3": {
    tr: "Gövdenizle değil, olduğunuz yerde dönerek çevirin. Adım atmak çekimi bozar.",
    en: "Turn on the spot rather than walking the phone around. A step is what ruins a capture.",
  },
  "capture.stand.ready": { tr: "Hazırım", en: "I'm ready" },

  // ── Framing gates ────────────────────────────────────────────────────────────────────
  "capture.frame.hold": { tr: "Sabit tutun", en: "Hold still" },
  "capture.frame.level": { tr: "Dikleştirin", en: "Straighten up" },
  "capture.frame.dark": { tr: "Işık az", en: "Too dark" },
  "capture.frame.bright": { tr: "Pencereden başlamayın", en: "Don't start on a window" },
  "capture.frame.turnhere": { tr: "Buradan başlayın", en: "Start here" },
  "capture.frame.starting": { tr: "Başlıyor", en: "Starting" },
  "capture.frame.force": { tr: "Yine de başla", en: "Start anyway" },
  "capture.frame.slowfps": {
    tr: "Kamera yavaş çalışıyor. Işıkları açarsanız kareler keskinleşir.",
    en: "The camera is running slow. Turning the lights on will sharpen the frames.",
  },

  // ── Rows ─────────────────────────────────────────────────────────────────────────────
  "capture.row.level": { tr: "Göz hizası", en: "Eye level" },
  "capture.row.up": { tr: "Yukarı sıra", en: "Upper row" },
  "capture.row.down": { tr: "Aşağı sıra", en: "Lower row" },
  "capture.cue.turn": { tr: "Yavaşça sağa dönün", en: "Turn slowly to your right" },
  "capture.cue.up": { tr: "Biraz yukarı bakın", en: "Tilt up a little" },
  "capture.cue.down": { tr: "Biraz aşağı bakın", en: "Tilt down a little" },
  "capture.cue.back": { tr: "Başladığınız yere dönün", en: "Come back round to the start" },
  "capture.row.done": { tr: "Sıra tamam", en: "Row complete" },
  // Shown over the first ring of the up and down rows while the phone settles. It has to
  // say what to DO, not what is happening: a bare number over a viewfinder reads as a
  // shutter timer, and the thing being asked for is stillness, not readiness.
  "capture.row.settle": { tr: "Sabit tutun", en: "Hold still" },
  // The row transition used to be a 1.2 s picture with no words, so which row came next was
  // something the user had to infer from a wireframe band. Naming it is the difference
  // between a guide and an instrument.
  "capture.row.next": { tr: "Sırada: {row}", en: "Next: {row}" },
  "capture.row.nextUp": { tr: "Telefonu yukarı eğin, sonra yine sağa dönün", en: "Tilt the phone up, then turn right again" },
  "capture.row.nextDown": { tr: "Telefonu aşağı eğin, sonra yine sağa dönün", en: "Tilt the phone down, then turn right again" },
  // A row ends on TWO conditions — the rings AND a closed 360 turn — and until now only the
  // rings were visible. A user who filled every ring and then stood still was watching a
  // screen that had stopped reacting, with nothing anywhere saying the turn was still open.
  // This is the cue for exactly that state.
  "capture.cue.closeLoop": {
    tr: "Halkalar tamam — turu kapatmak için başladığınız yere dönün",
    en: "Rings done — turn back to where you started to close the loop",
  },

  // ── Warnings ─────────────────────────────────────────────────────────────────────────
  // One verb, then one clause that names the fix. WARNING_COPY in the contract maps each
  // WarningCode onto these keys, so the wording and the detector can never drift apart.
  "capture.warn.walk": { tr: "Yürümeyin", en: "Don't walk" },
  "capture.warn.walk.body": {
    tr: "Ayaklarınız sabit kalsın, sadece telefon dönsün.",
    en: "Keep your feet planted; only the phone turns.",
  },
  "capture.warn.fast": { tr: "Yavaşlayın", en: "Slow down" },
  "capture.warn.fast.body": { tr: "Bir tur yaklaşık 15 saniye.", en: "One full turn takes about 15 seconds." },
  "capture.warn.smear": { tr: "Daha yavaş", en: "Slower still" },
  "capture.warn.smear.body": {
    tr: "Işık az; hızlı dönünce kareler bulanıyor.",
    en: "The light is low; turning fast smears the frames.",
  },
  "capture.warn.pivot": { tr: "Dirsekleri toplayın", en: "Elbows in" },
  "capture.warn.pivot.body": {
    tr: "Telefon gövdenizin önünde, size yakın dönsün.",
    en: "Keep the phone close in front of your chest as it turns.",
  },
  "capture.warn.glare": { tr: "Fazla parlak", en: "Too bright" },
  "capture.warn.dark": { tr: "Çok karanlık", en: "Too dark" },
  "capture.warn.tilt": { tr: "Düz tutun", en: "Keep it level" },
  "capture.warn.rowpitch": { tr: "Hizaya dönün", en: "Back to the row" },
  "capture.warn.lowfps": { tr: "Kare düşüyor", en: "Frames dropping" },
  "capture.warn.lens": { tr: "Lens değişti", en: "Lens changed" },
  "capture.warn.lens.body": {
    tr: "Yakınlaştırmaya dokunmayın; motor tek bir lense göre çözüyor.",
    en: "Don't touch the zoom; the engine solves for one lens only.",
  },
  "capture.warn.lost": { tr: "Yön kayboldu", en: "Heading lost" },

  // ── Controls and pause ───────────────────────────────────────────────────────────────
  "capture.finish": { tr: "Bitir", en: "Finish" },
  "capture.pause.title": { tr: "Duraklatıldı", en: "Paused" },
  "capture.pause.body": {
    tr: "Kayıt bekliyor. Şimdiye kadar topladığınız kareler duruyor.",
    en: "Recording is on hold. Everything you have banked so far is safe.",
  },
  "capture.pause.resume": { tr: "Devam et", en: "Resume" },
  "capture.pause.finish": { tr: "Çekimi bitir", en: "Finish the capture" },
  "capture.pause.restart": { tr: "Baştan başla", en: "Start over" },
  "capture.resume.aim": { tr: "Parlak halkayı bulun ve bekleyin", en: "Find the bright ring and hold" },
  "capture.early.title": { tr: "Şimdi bitirilsin mi?", en: "Finish here?" },
  "capture.early.body": {
    tr: "{n} kare eksik. Eksik kalan yerleri motor tahmin etmek zorunda kalır.",
    en: "{n} tiles still missing. Whatever is left, the engine has to guess.",
  },
  "capture.band.ceiling": { tr: "tavan", en: "the ceiling" },
  "capture.band.floor": { tr: "zemin", en: "the floor" },

  // ── Sweep check and review ───────────────────────────────────────────────────────────
  "capture.check.working": { tr: "Ölçülüyor…", en: "Measuring…" },
  "capture.review.great.title": { tr: "Çekim tam", en: "That's a full capture" },
  "capture.review.great.body": {
    tr: "Kürenin %{n}'i kapsandı — bu çekimden temiz bir panorama çıkar.",
    en: "{n}% of the sphere is covered — this will stitch cleanly.",
  },
  "capture.review.gaps.title": { tr: "Eksik bölgeler var", en: "There are gaps" },
  "capture.review.gaps.body": {
    tr: "Kürenin %{n}'i kapsandı. Eksikleri tamamlamak bir kaç saniye sürer ve sonucu belirgin şekilde düzeltir.",
    en: "{n}% of the sphere is covered. Filling the gaps takes seconds and visibly improves the result.",
  },
  "capture.review.walk.title": { tr: "Kamera yer değiştirmiş", en: "The camera moved" },
  "capture.review.walk.body": {
    tr: "Çekim sırasında adım atılmış. Motor böyle çekimleri geri çevirebiliyor — olduğunuz yerde dönerek tekrar çekmek en güvenlisi.",
    en: "There were steps during the sweep. The engine refuses captures like this — the safest fix is to shoot it again turning on the spot.",
  },
  "capture.review.dark.title": { tr: "Görüntü karanlık", en: "The footage is dark" },
  "capture.review.dark.body": {
    tr: "Kareler bulanık çıkmış olabilir. Işıkları açıp tekrar çekerseniz detay belirgin şekilde artar.",
    en: "The frames are probably smeared. Turn the lights on and shoot again for a visible jump in detail.",
  },
  "capture.review.fill": { tr: "Eksikleri tamamla", en: "Fill the gaps" },
  "capture.review.use": { tr: "Yükle ve işle", en: "Upload and process" },
  "capture.review.retake": { tr: "Baştan çek", en: "Shoot again" },
  "capture.review.anyway": { tr: "Yine de yükle", en: "Upload anyway" },
  "capture.review.thumbs": { tr: "Çekimden kareler", en: "Frames from the capture" },
  "capture.review.enough": { tr: "Yeterli sınır", en: "Good-enough line" },

  // ── Bonus, upload, draft ─────────────────────────────────────────────────────────────
  "capture.bonus.poles": { tr: "Tavanı ve zemini de çek", en: "Also shoot ceiling and floor" },
  "capture.bonus.skip": { tr: "Gerek yok", en: "Not needed" },
  "capture.save.local": { tr: "Videoyu telefona kaydet", en: "Save the video to this phone" },
  "capture.upload.title": { tr: "Yükleniyor", en: "Uploading" },
  "capture.upload.retry": { tr: "Tekrar dene", en: "Try again" },
  "capture.upload.slowlink": {
    tr: "Bağlantı yavaş görünüyor. {n} MB'lık yükleme için Wi-Fi'ye geçmek işi kısaltır.",
    en: "The connection looks slow. Switching to Wi-Fi will shorten this {n} MB upload.",
  },
  "capture.draft.title": { tr: "Yarım kalmış çekim", en: "An unfinished capture" },
  "capture.draft.body": {
    tr: "Önceki çekim tamamlanmadan kesilmiş. Kaldığı yerden devam edebilir ya da baştan başlayabilirsiniz.",
    en: "The last capture was cut short. You can carry on from where it stopped, or start over.",
  },

  // ── Settings and accessibility labels ────────────────────────────────────────────────
  "capture.voice": { tr: "Sesli rehber", en: "Voice guide" },
  "capture.sound": { tr: "Ses", en: "Sound" },
  "capture.progress.label": { tr: "Çekim ilerlemesi", en: "Capture progress" },
  "capture.progress.value": { tr: "%{n} tamamlandı", en: "{n}% complete" },
  "capture.tiles.remaining": { tr: "{n} kare kaldı", en: "{n} tiles to go" },

  // ── Spoken lines ─────────────────────────────────────────────────────────────────────
  // Spoken through SpeechSynthesis, so they are full sentences with punctuation the TTS
  // engine can use for prosody — unlike the on-screen copy, which is deliberately clipped.
  "capture.say.start": { tr: "Yavaşça sağa dönmeye başlayın.", en: "Start turning slowly to your right." },
  "capture.say.up": { tr: "Şimdi biraz yukarı bakın ve dönmeye devam edin.", en: "Now tilt up a little and keep turning." },
  "capture.say.down": { tr: "Şimdi biraz aşağı bakın ve dönmeye devam edin.", en: "Now tilt down a little and keep turning." },
  "capture.say.slow": { tr: "Biraz yavaşlayın.", en: "Slow down a little." },
  "capture.say.walk": { tr: "Olduğunuz yerde durun.", en: "Stop where you are." },
  "capture.say.gap": { tr: "{band} tarafında eksik kaldı.", en: "There's a gap towards {band}." },
  "capture.say.done": { tr: "Çekim tamam.", en: "Capture complete." },
} satisfies Dict;

/** Values the `{n}` / `{band}` placeholders accept. */
export type CopyVars = Record<string, string | number>;

/**
 * Look a capture string up.
 *
 * `vars` is optional and additive to the shape the contract specifies (`cc(key, locale)`),
 * so callers that need no interpolation read exactly as documented. Substitution is a
 * literal `{name}` replace rather than a template so that the dictionary stays plain data
 * — translators never see code.
 */
export function cc(key: CaptureCopyKey, locale: Locale = "tr", vars?: CopyVars): string {
  const text = captureStrings[key][locale];
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (whole, name: string) => {
    const value = vars[name];
    // An unknown placeholder is left verbatim: a visible `{n}` in QA is far cheaper than a
    // silent "undefined" in front of a customer.
    return value === undefined ? whole : String(value);
  });
}

/** Row heading, e.g. "Göz hizası". Kept here so P never string-builds a row name. */
export function rowTitleKey(row: "level" | "up" | "down"): CaptureCopyKey {
  return row === "level" ? "capture.row.level" : row === "up" ? "capture.row.up" : "capture.row.down";
}

/** The cue spoken and shown for a row while it is being swept. */
export function rowCueKey(row: "level" | "up" | "down"): CaptureCopyKey {
  return row === "level" ? "capture.cue.turn" : row === "up" ? "capture.cue.up" : "capture.cue.down";
}

/** The line the voice guide speaks when a row begins. */
export function rowVoiceKey(row: "level" | "up" | "down"): CaptureCopyKey {
  return row === "level" ? "capture.say.start" : row === "up" ? "capture.say.up" : "capture.say.down";
}
