# BioSphere AI — Mimari ve Algoritma

Bu belge, "telefon videosu → kusursuz 360° panorama" iddiasının arkasındaki mühendislik kararlarını açıklar. Kod okunurken *ne* yapıldığı bellidir; burada *neden* öyle yapıldığı yazılıdır.

---

## 1. Koordinat sistemi

Panorama işlerinde hataların çoğu bir eksik transpozeden veya bir ters işaretten çıkar. Bu yüzden tek bir sözleşme var ve tamamı `engine/biosphere/pano/geometry.py` içinde uygulanmıştır. Başka hiçbir dosya bu matematiği yeniden türetmez.

| | |
|---|---|
| **Dünya** | Sağ el, **+X sağ, +Y aşağı, +Z ileri** |
| **Kamera** | +X sağ, +Y aşağı, +Z ileri (OpenCV standardı) |
| **`R`** | Dünya yönünü kamera yönüne çevirir: `p_kamera = R @ d_dünya` |
| **Equirect** | `θ = 2π(u+0.5)/W − π`, `φ = π/2 − π(v+0.5)/H` |
| **Işın** | `(cos φ·sin θ, −sin φ, cos φ·cos θ)` |

**Neden +Y aşağı?** Dünya ekseni kamera ekseniyle özdeş olsun diye. Böylece dönmemiş bir kameranın rotasyonu tam olarak birim matristir. "Y yukarı" seçseydik her yerde `diag(1,−1,1)` taşımak zorunda kalırdık ve o çarpanın unutulduğu tek bir yer panoramayı baş aşağı çevirirdi.

Satır 0 zenit, son satır nadir, sütun 0 döngüsel dikiş yeridir. `tests/test_geometry.py` bu sözleşmenin tamamını sabitler; yön hataları oraya çarpar, panoramaya değil.

---

## 2. Neden saf rotasyon?

Kullanıcı yerinde döner; kamera merkezi *yaklaşık* sabittir. Bu, problemi kamera başına 6 serbestlik derecesinden (SfM) 3'e indirir.

Bunun pratikteki karşılığı büyüktür:

- **Koşullanma.** Öteleme + derinlik çözmeye çalışmak, düz beyaz bir duvara bakarken dejenere bir problemdir. Sadece rotasyon çözmek değildir.
- **Homografi geçerli olur.** İki görüntü arasındaki eşleme tam olarak `H = K R K⁻¹`'dir. Sahne derinliğinden bağımsız olarak. Bu, hem eşleştirmeyi doğrulamayı hem de odak uzaklığını homografiden kestirmeyi mümkün kılar.
- **Küre kapanabilir.** 360° dönüşün başı ve sonu aynı yöne bakar; bu, optimizasyona doğrudan yazılabilen bir kısıttır.

Bedeli paralakstır ve bu bedel gerçektir: telefon, kolun/omurganın ucunda ~25–30 cm yarıçaplı bir yay çizer. Yakın nesneler ile uzak nesneler kareler arasında farklı miktarda kayar. Hiçbir rotasyon modeli bunu düzeltemez — bu yüzden `refine.py` var (bkz. §7).

---

## 3. Kare ayıklama ve seçimi

Bir video, panorama için gereğinden çok fazla kare içerir ve bunların çoğu kullanılamaz.

1. **Ayıklama.** `ffmpeg` ile hedef fps'te, Lanczos ile ölçeklenerek. `-noautorotate` **kullanılmaz**: telefon videolarındaki `displaymatrix` metadata'sı uygulanmalıdır, yoksa dikey çekilmiş video yan yatar.
2. **Kalite skoru.** Laplacian varyansı (keskinlik), ortalama parlaklık, kırpılmış piksel oranı.
3. **Hareket kestirimi.** 360 px'e küçültülmüş gri kareler arasında `cv2.phaseCorrelate`. Ucuz ve bulanıklığa dayanıklı — asıl öznitelik eşleştirmesinden *önce* kaba bir yaw izi verir.
4. **Anahtar kare seçimi.** Kümülatif yaw izi üzerinde `hfov · (1 − örtüşme)` adımlarla ilerlenir ve **her açısal kutunun en keskin karesi** seçilir — en yakın karesi değil. Bulanık bir kareyi almak, o bölgede kalıcı bir yumuşaklık bırakır.

Hedef örtüşme %55'tir. Daha az örtüşme daha az kare demektir ama bundle adjustment'ın koşullanmasını bozar; daha fazlası kareyi çoğaltır ve harmanlamayı yavaşlatır.

---

## 4. Öznitelik eşleştirme

- **SIFT**, `contrastThreshold=0.015` ile. Varsayılan eşik iç mekan duvarları için fazla katıdır; düşük eşik gürültülü öznitelikler üretir ama RANSAC onları temizler. Az öznitelik bulmak, kötü öznitelik bulmaktan daha tehlikelidir.
- **RootSIFT** normalizasyonu (L1 + karekök). Bedava ve eşleştirme kalitesini ölçülebilir biçimde artırır.
- **CLAHE** ön işlemesi, düz duvarlardaki zayıf dokuyu öne çıkarır.
- Geometrik doğrulama `cv2.findHomography(..., USAC_MAGSAC)` ile. Dejenere homografiler (kolineer noktalar, aşırı kesme) reddedilir: saf rotasyon altında yakın görüntüler arasında o kadar kesme olamaz.

**Aday çiftler:** ardışık komşular her zaman; ayrıca kaba yaw izine göre örtüşmesi beklenen tüm çiftler — **360° sarmalanan çift dahil**. Bu son madde loop closure'ın tek kaynağıdır.

**Dokusuz boşluklar köprülenir.** Hiçbir kareyle doğrulanmış öznitelik paylaşmayan bir anahtar kare — boş duvar, çıplak tavan köşesi — çözümden düşer (`_keep_largest_component`) ve o yön render'da komşu satırın çerçeve kenarıyla, yani en kötü pikselleriyle örtülürdü. Referans çekimde bu, üst satırda 62°'lik bir delikti: 421–467 numaralı kaynak kareler yalnızca düz duvar ve korniş görüyor, SIFT aralarında — çeyrek saniye arayla çekilmiş ardışık karelerde bile — tek bir çift doğrulayamıyor (ham eşleşme 0–5) ve köşede korniş yalnızca seviye satırın üst kenarından gelip yıkanıyordu. Kareler var ve keskin; sadece tanımlayıcının tutunacağı bir şey taşımıyorlar. `solve._bridge_track_gaps` bunları süpürmede kendilerini çevreleyen çözülmüş anahtar kareler arasında **interpolasyonla** yerleştirir — uç noktalar kesindir, yalnız boşluk içindeki oran tahmindir — ve **yalnızca render'a** ekler; çözüme, döngü kapanışına ve raporlanan hatalara dokunmaz. Bir boşluğun satır **içinde** mi yoksa satırlar **arasında** mı olduğu, sabit bir pitch adımıyla değil boşluğun yönüyle ayrılır: bir satır yaw boyunca ilerler, satır geçişi pitch boyunca. Görünüşte doğru olan iki ölçüt de ölçümde yanlış çıktı. Sabit 12°'lik pitch adımı, üst satırdaki gerçek bir 49°'lik deliği reddetti — çünkü elle çevrilen satır düz değildir ve o deliğin iki ucu +33° ile +19°'de, yani 13.9° ayrıydı; yanındaki gerçek satır geçişi ise 26.2°. Pozları pitch'e göre kümelemek (render'ın `_pitch_rows` yaptığı gibi) bu çekimde üç satır yerine **tek** satır buluyor: operatör satırlar arası eğilirken de kaydettiği için pitch dağılımı -36°'den +34°'ye kesintisiz ve bölünecek boş bant yok. Yön ikisinden de sağ çıkıyor. Oran, iz boşluğu gördüyse hareket izinden, görmediyse zamandan gelir: düz duvarda faz korelasyonu kornişe kilitlenir, bu pitch'i sabitler ama yaw'ı değil (açıklık problemi; referans çekimde çözümün 62° dediği boşlukta iz 47° ölçtü). Yatay bir korniş aynı sebeple birkaç derecelik yaw hatasını gizler: kare, hizalandığı çizgi boyunca kayar. Kalan birkaç derecelik hata render'da giderilir, iki katmanla. Önce hiç seyrek kısıtı olmayan bir görüntü, yoğun geçişten önce rijit olarak **yeniden nişanlanır** (`render._reaim_views`): öneri, görüntünün komşularının bileşkesine **ECC görüntü hizalamasıyla** (öteleme modeli, kabadan inceye) gelir — her pikseli, gölgelemeyi de kullanır; blok eşleştiricinin medyanı düz duvarda gürültüdür (ölçüldü: 8192 px'te köprülenen bir görüntüyü korniş boyunca 9° yürütüp ikiledi ve 4096 ile 8192'de farklı cevap verdi; ECC iki genişlikte de aynı 0.5–0.8°'lik düzeltmeleri önerdi). Yargı, yoğun geçişin kendi cetvelidir: penceresinde (10 px) eşleyebildiği blok sayısı ve rezidüsü; iyileşmeyen hamle geri alınır, 5°'lik tavanı aşan eksen (kenar boyunca kaymıştır: kornişi olan bir görüntü için ECC 16.5° yaw önerdi) düşürülür, iki yönde de reddedilen görüntü interpolasyonun koyduğu yerde kalır. Sonra yoğun geçişte bu görüntülere dokunan çiftlerin arama penceresi 10 px'ten 5°'ye açılır ki mesh onlara ulaşabilsin — yalnız o çiftlerin: pencereyi her çift için 24 px'e açmak düz duvar gürültüsünü her yere soktu (mesh p90 3.2 → 9.6 px, çizgi düzlüğü 3.91 → 4.23). Ölçüm (4096, gerçek çekim): köprülenen görüntülerden ikisi ECC ile 0.45–0.83° yaw ve 0.44–0.54° pitch döndü (yoğun pencere blokları 218 → 310, rezidü 10.0 → 6.8 px); üçüncüsünün yaw'ı düz duvarda gözlemlenemez ve mesh'in geniş penceresine bırakıldı. Korniş köşeden kesintisiz geçiyor; çizgi düzlüğü 3.91 → 3.58 px, mesh p90 3.2 → 3.7 px (geniş çiftlerin bedeli). Kapsama 0.835 → 0.849, "1 kapsama boşluğu" uyarısı kalktı, köşedeki yıkanmış bant gitti ve korniş köşeden kesintisiz geçiyor (çizgi düzlüğü 3.91 → 3.23 px; nişanlama olmadan köprülenen kareler kornişe 15–20 px'lik basamaklar koyuyordu).

---

## 5. Bundle adjustment ve loop closure

Zincirleme (spanning tree) ile başlangıç pozları kurulur, sonra global optimizasyon:

- **Parametreler:** kamera başına yerel bir dönüş sapması (`R ← exp(δ) R`, kamera 0 sabit) + paylaşılan `log(odak)` + isteğe bağlı `k1, k2`.
- **Rezidü:** her eşleşme için, i'deki noktayı ışına çevir → dünyaya döndür → j'ye yansıt → piksel farkı. Çift yönlü.
- **Kayıp:** Huber. Aykırı değerler kaçınılmazdır ve karesel kayıp onlara teslim olur.
- **Kendi Levenberg–Marquardt'ımız, analitik Jacobian ile.** Eskiden SciPy'nin trust-region çözücüsüne sonlu farklarla Jacobian veriliyordu ve onun `lsmr` alt çözücüsü varsayılan toleranslarıyla *kesin olmayan* adımlar atıyordu: 89 kameralık gerçek çekimde her tur ~385 iterasyonla saatin sınırına çarptı, çözüm 289–341 sn sürdü ve durduğu yere göre farklı bir odak verdi. Parametre sayısı küçüktür (kamera başına 3 + odak), normal matris birkaç yüz boyutundadır; pahalı olan Jacobian'dır ve o kapalı formdadır. Aynı çekim şimdi ~5 sn'de yakınsıyor.
- **Önce rotasyon ortalaması.** Her çiftin en iyi salt-rotasyon uyumu (ışınlar üzerinde Kabsch) bir göreli dönüş ölçümüne dönüştürülür ve döngü kenarları dahil bütün graf üzerinde ortalanır. Spanning tree kapanış kenarlarını yapı gereği görmezden gelir; ortalama hepsini kullanır ve piksel çözümüne döngüye tutarlı bir başlangıç verir. `pair.R` de artık homografiden değil bu uyumdan gelir: el çekiminde homografi paralaksı projektif terimlerine emer ve ondan çıkarılan rotasyon onlarca derece sapabilir (satırlar arası çiftlerde 64–142 px rezidü ölçüldü, doğrudan uyumda 17–32 px).
- **Veriye göre seçilen atma eşiği.** Son tur rezidüsü `max(3·f_scale, 3·1.4826·MAD)` üstündeki eşleşmeleri atar ve yeniden uydurur. Sabit 6 px eşik, el çekiminde eşleşmelerin %54–61'ini — bütün yakın alanı — atıp uyumu uzak duvara demirliyordu. Hem sağ kalanların hem de **bütün** eşleşmelerin istatistikleri raporlanır; ikincisi (`parallax_median_px`, `parallax_p90_px`, `parallax_fraction_over_6px`) çekimin bir rotasyondan ne kadar uzak olduğunu söyler, birincisi bunu gizler.

**Loop closure** ayrı bir adım değildir: sarmalanan çift grafikte bir kısıt olarak durduğu için optimizasyon sürüklenmeyi kendiliğinden tüm tur boyunca dağıtır. Ölçülen büyüklük `loop_closure_residual_deg`'dir; sağlıklı bir çekimde BA sonrası 1°'nin altındadır. Paralakslı bir çekimde ise bu sayı bir *tutarsızlık* ölçüsüdür, çözücünün hatası değil: satırlar birbirine göre gerçekten kaymıştır ve hiçbir rotasyon seti hepsini birden memnun edemez.

**Ölçülüp reddedilenler.** Faz korelasyonu yaw izini `cos(pitch)` ile düzeltmek (eğik satır bir turu daha az pikselde görür) kâğıt üstünde doğru, ölçümde yanlıştı: düz iz, eşleştiricinin sonradan doğruladığı 307 çiftin hepsini önerirken düzeltilmiş iz 89'unu — satırlar arası çiftlerin tamamını — kaçırıyor ve bütün alt satırı (34 anahtar kare) kopuk bırakıyordu. Faz korelasyonunun pitch'i de satır geçişlerinde ~10° sapıyor. Seçenek `motion_to_angles(pitch_correction=...)` olarak kaldı, varsayılanı kapalı. Aday çiftlerde pitch farkı `0.8·vfov` üstünde olanların elenmesi ise ölçümde zararsız (307/307 hâlâ öneriliyor, aday sayısı %20 az).

**Kol paralaksı odağı belirlenemez kılar — ve bu çözücünün hatası değildir.** Telefon 28 cm'lik bir kolun ucunda dönerken teğetsel öteleme, rotasyon akışıyla **aynı yönde** `f·r/d` kadar ek görüntü akışı üretir (`d` sahne derinliği). Salt rotasyon modeli 360°'lik döngüyü ancak odağı yaklaşık `(1 + r/d)` kadar uzatarak kapatabilir; derinliğe bağlı kalan kısmı da yaw aralıklarına dağıtır. Sentetik tek satırlı fikstürde (gerçek odak 579.4 px, 45°) ölçüldü: bundle her başlangıçtan (520, 579.4, 640, 700 px) aynı 655.4 px'e (%13 uzun, 40.2°) yakınsıyor; sabit odaklı maliyet profili 600–670 px arasında %8 içinde düz. Çift başına 2-B öteleme nüsansı eklemek (`predicted += f·τ_ij`, sıfıra doğru öncüllü) cevabı yalnızca üçte bir yol geri getiriyor (629.7 px), öncülün gücünden bağımsız olarak: uyan `|τ|·f` medyanı 3.6 px, beklenen ~50 px'lik kol paralaksının onda biri — paralaksı emen nüsans değil, rotasyonlar artı odak. Derinlik olmadan ayrıştırılamaz; tam SfM gerekir ve o bu ürünün modeli değildir. Sonuç, mozaik için *doğru* olan kendi içinde tutarlı efektif odaktır: dikişler kapanır, geometri yaklaşık `r/d` oranında sıkışır (oda olduğundan biraz büyük görünür). Gerçek referans çekimde de aynı iz var: ön-tarama 50.3° ölçtü, çözüm 45.2°'ye uzattı. Bu yüzden kalite kapısı odak hatasını yalnızca **nodal** senaryoda (paralakssız; ölçülen hata 0.009°) çözücüye yükler; kollu senaryoda efektif alanın açıklanan aralıkta ([38°, 46°]) kaldığını ve poz hatasının derinlik dağılımının imâ ettiği büyüklükte olduğunu ister. Sürüm çözücüsü aynı fikstürde 652.2 px ve 1.71° ortalama poz hatası veriyordu, 154 sn'de; şimdiki 655.4 px, 1.52°, 15 sn.

**Telefonun objektifi bir donanım sabitidir, ve artık okunuyor.** Tarayıcı bir kameraya görüş açısını soramaz: `MediaTrackSettings` genişlik, yükseklik, kare hızı, zoom, ISO ve pozlama süresi taşır, optik hakkında hiçbir şey taşımaz (`focalLength` hiçbir tarayıcıda uygulanmadı). Bu yüzden yakalama uygulaması sidecar'a varsayılan bir `optics.hfovDeg` yazıyor, motor da onu görmezden gelip odağı görüntüden ölçüyordu. Ölçmek doğru içgüdü ama yetmiyor, çünkü **ölçüm tek yönlü sapıyor**: el çekimi omuz etrafında döner, teğetsel öteleme rotasyon akışıyla aynı yönde akış ekler ve salt rotasyon modeli 360°'yi ancak odağı yaklaşık `1 + r/d` kadar **uzatarak** kapatabilir. Kısa çıkamaz. Aynı telefonla (vivo Y22s) aynı odanın iki çekimi: geniş kollu çekimde 45.2°, yavaş ve dar kollu çekimde 51.7° — hiç değişmeyen bir objektiften 6.5° fark.

`devices.py` telefonu adlandırıp objektifi tablodan okur. **Android** modeli doğrudan verir: `navigator.userAgentData.getHighEntropyValues(["model"])` üreticinin SKU'sunu döndürür (vivo Y22s → `V2206`, Galaxy S23 → `SM-S911B`); UA dizesi işe yaramaz, Chrome onu 2022'de "Android 10; K" olarak dondurdu — referans çekimin sidecar'ında yazan tam olarak budur. **iOS** modeli hiçbir API'den vermez; geriye ekran kalır: CSS nokta ölçüsü ve piksel oranı iPhone'ları küçük ailelere ayırır. Aile bir telefon değildir ve her zaman hemfikir de değildir — 393×852@3 hem 24 mm'lik 14 Pro/15 Pro'yu hem 26 mm'lik 15/16'yı kapsar — bu yüzden parmak izi ailenin **ortasını ve yayılımını** taşır. **Hiçbiri** tanınmazsa bile durum eskisinden iyidir: 2018'den beri her telefon ana kamerası 23–27 mm arasındadır, jenerik profil 26 mm der ve bu 4:3 dikey karede 53°'dir — motorun hiçbir şey bilmediğinde varsaydığı 67°'ye göre 1.5 kat daha doğru.

Dönüşüm tek satırdır ve iki tuzağı vardır. 35 mm eşdeğeri **köşegen** üzerinde tanımlıdır (CIPA, 43.267 mm), dolayısıyla `focal_px = f35 · köşegen_px / 43.267` ve kare oranı denkleme hiç girmez. Birinci tuzak: ölçülmüş bir yatay görüş açısından 36 mm **genişlik** ile hesaplamak %8 hata verir — tam olarak giderilmeye çalışılan sapmanın büyüklüğü. İkinci tuzak: üreticiler fotoğraf karesini kote eder, 16:9 video onun kırpılmışıdır ve ~1.1 çarpanı ister; uygulama 1080×1440, yani sensörün kendi 4:3'ünü kaydettiği için kırpma yoktur ve fotoğraf değeri doğrudur. Referans cihazda doğrulandı: Camera2'nin bildirdiği 65.7° yatay alan köşegen dönüşümünde 26.8 mm, 3:4 karede 51.7° eder; motorun yavaş çekimdeki kendi öz-kalibrasyonu 51.65° buldu — 0.03° fark.

**Ve bu bir çit, hedef değil.** Kolun 0.28 m olduğu, gerçek odağın 579.4 px bilindiği sentetik çekimde ölçüldü: odağı **gerçek değerine sabitlemek panoramayı kötüleştirdi** (ufuk PSNR 18.52 → 17.18 dB, ortalama poz hatası 1.52 → 2.51°). Gerçek paralaks altında geometrik olarak doğru odak, en iyi mozaik odağı değildir — şişmiş odak, yer değiştiren bir kameranın daha iyi bir yaklaşımıdır. Serbest bırakmak da en iyisi değil (+%13 sapma). En iyisi ikisinin arası: σ=0.06 ile 2σ'lık bir bant (PSNR 18.77, poz 1.28, döngü kalıntısı 0.153 → 0.083). Daha dar bantlar sarma sütununu kendi kapısının üstüne itti (2.07–2.19, sınır 2.0). Gerçek üç satırlı çekimde aynı bant odağı 1286 → 1195 px'e çekti ve döngü kalıntısını 3.47° → 1.36° düşürdü — görsel denetimin 1 numaralı kusuru (kapı üstünde beş kez çizilen korniş) tam olarak oradaki tutarsızlıktı. Bant `bundle._robust_lm` içinde her LM adımında uygulanır; σ profilin kendi belirsizliğidir (ölçülmüşse 0.06, üretici yayınlamışsa 0.08, tahminse 0.12) ve sanal çoklu-kamera cihazlarında gevşetilir.

Motor ayrıca çözümle objektifin **anlaşmazlığını** raporlar: solve objektiften 3σ'dan fazla uzunsa bu paralaksın ölçüsüdür (uyarı verilir, alan objektifin toleransında tutulur); 3σ'dan fazla **kısaysa** bu paralaksla açıklanamaz ve muhtemelen başka bir kamera seçilmiştir. LiDAR'lı bir model tanındığında da tek bir cümle söylenir: tarayıcı LiDAR'a erişemez (Safari'de WebXR yok, derinlik API'si yok, ImageCapture yok), bu yüzden ölçüm herkesle aynı yoldan — varsayılan kamera yüksekliğinden — gelir.

**Ufuk düzleme.** Her kameranın "aşağı" vektörü dünya koordinatlarında yerçekimi ekseni etrafında kümelenir. Bu vektörlerin SVD'si ekseni verir; dünya ona göre döndürülünce ufuk ekvatora oturur. Telefonun nasıl tutulduğu artık önemsizdir.

---

## 6. Fotometrik denge

Telefon otomatik pozlaması tur boyunca sürüklenir; pencereye bakınca kararır, duvara bakınca açılır. Düzeltilmezse panoramada bant bant parlaklık farkı görünür.

- **Kazanç dengeleme (Brown & Lowe).** Örtüşme başına ortalama yoğunluklar üzerinden kurulan doğrusal sistem, kamera başına kanal bazlı kazanç verir. Regülarizasyon terimi tüm panoramanın topluca kararmasını engeller.
- **Vinyetleme.** Örtüşme çiftlerinde, aynı sahne noktası iki farklı görüntüde farklı yarıçaplarda düşer. Bu oranlardan paylaşılan radyal bir model (`1 + a₁r² + a₂r⁴ + a₃r⁶`) uydurulur. Model [0.5, 1.3] aralığına kelepçelenir — çılgın bir uyum, hiç düzeltme yapmamaktan kötüdür.
- **İkisi de gürbüz.** Piksel-piksel log-oranları hizalama hatası ve paralaksla doludur (gerçek çekimde radyal modelin açıkladığı varyans %0.3, örneklerin %12.5'i kaba aykırı); düz en küçük kareler buradan *tekdüze olmayan* bir eğri çıkarıp her kareye bir halka basıyordu. Uyum Huber-IRLS ile yapılır, log-uzayında uydurulan eğri uygulanan `1 + poly` modeline dönüştürülür (güçlü bir vinyette ikisi köşede %15 ayrışır) ve köşeye doğru tekdüze azalmayan bir uyum reddedilir. Kazanç sisteminde de çift ağırlıkları her adımda log-oran rezidüsüne göre yeniden verilir: bir lamba ya da pencereyi kesen birkaç çift artık komşularının kazancını %10 kaydıramaz.

---

## 7. Render: sarmalama, dikiş, harmanlama

### Sarmalama problemi

Equirect tuval yatayda döngüseldir ama OpenCV'nin dikiş bulucusu ve harmanlayıcısı düz dikdörtgen varsayar. Çözüm, kütüphaneyle savaşmak yerine **dünyayı döndürmek**:

> Tuval kenarı (u = 0), **tek bir kaynak görüntünün footprint'inin içine** düşecek şekilde global bir yaw uygulanır.

Sonuç: kenarın iki yanındaki pikseller aynı görüntüden, aynı warp ile gelir. Süreklilik yapı gereği sağlanır. Dikiş bulucu ve harmanlayıcı döngüsellikle hiç ilgilenmez. Ek bir güvence olarak, bu görüntü W/2 kaydırılmış bir tuvale bir kez daha render edilip u = 0 çevresinde 24 px'lik bir bant çapraz geçişle yamanır.

Seçim kuralı **satır başına** işler ve sırası ölçülmüştür: önce her satırın bandı tutan bir kamerası olsun (üç satırlı çekimde "en az kamera" tek başına, üst satırda delik olan bir yaw'ı seçip kenardaki tavanı sentezletiyordu), sonra **en az kamera**, en son en derin marj. Derinliği en az kameranın önüne koymak ölçümde yanlış çıktı: tek satırlı fikstürde üç kameranın paylaştığı bir yaw seçti ve sarma sütunu sahnenin kendi kenarlarının 1.83 katı öne çıktı; tek kameralık yaw 0.84 (mesh kapalı, yalnız yerleşim değişiyor). Mesh'le birlikte iki düzeltme daha: kenarı saran görüntünün tam genişlikteki ızgarası kısıtlarını pencere **merkezine** göre açar (sol kenara göre açmak `u ≈ W`'deki yarısını negatif `u`'ya düşürüp atıyordu — görüntü yalnızca bir yanından hizalanıyordu; gerçek çekimde +4.600 kısıt, mesh sonrası p90 3.51 → 3.08 px) ve silindirin ilk/son sütunu çözümden sonra tam eşitlenir (regülarizasyonun ağırlıklı bağı pikselin kesrini açık bırakıyordu ve o kesir tam sarma sütununda görünüyordu). Fikstürde sarma sütunu 3.26 → 0.67; gerçek çekimde sarma artık düz duvara düşüyor ve komşu sütunlardan ayırt edilemiyor (0.26 vs 0.3–0.4).

### Yerel mesh hizalama (paralaks)

El çekimi merceğin etrafında dönmez. Referans üç satırlı çekimde en iyi *salt rotasyon* hizalaması doğrulanmış eşleşmelerde medyan 4 px, satırlar arasında 10 px, döngüyü kapatan çiftlerde 30 px'e kadar uyuşmazlık bırakıyor — hiçbir rotasyon seçiminin gideremeyeceği artık paralaks. Tuvalde bunun karşılığı çift çizilen dolap çıtaları, bulanan sandalye ve satırlar arasında inip çıkan korniş çizgisiydi.

`mesh.py` bunu **yerel olarak** çözer: her görüntü, equirect tuval üzerinde kaba bir yer değiştirme ızgarası alır (hücre 2.8°) ve bütün görüntülerin ızgaraları **birlikte** çözülür; örtüşen görüntülerin eşleşen içeriği aynı tuval pikseline düşsün diye. Üç terim yarışır: **veri** (eşleşme; Huber, çünkü bir metredeki sandalye ayağı arkasındaki duvarla anlaşamaz ve çoğunluk kazanmalıdır), **öncül** (her köşe sıfır yer değiştirmeyi tercih eder, örtüşmesi olmayan bölge rotasyonun koyduğu yerde kalır) ve **sertlik** (birinci ve ikinci farklar; yerel olarak afin bir warp düz çizgiyi düz bırakır — `refine.py`'nin optik akışının geometriyi bükmesinin sebebi tam da bu terimin yokluğuydu).

İki geçiş. İlki çözücünün doğrulanmış SIFT eşleşmelerini kullanır: kesin ama yalnızca köşelerin olduğu yerde. İkincisi görüntüleri o çözümle 2048 px'te yeniden warp eder ve her örtüşmede blok-blok normalize çapraz korelasyon yapar; korniş, süpürgelik ve düz kapı kasası gibi SIFT'in görmediği şeyleri görür ve her bloğu **kenara duyarlı** bir kısıta çevirir (yapı tensörüne göre yalnız kenarı dik kesen bileşen; kenar boyunca kayma açıklık problemidir ve gürültüden başka bir şey eklemez). Kenar blokları dokulu bloklara göre 3× ağırlık taşır: gözün ilk okuduğu mimari onlardır ve yakındaki bir nesnenin dokusu tarafından sayıca ezilirler. Her şey köşe yer değiştirmelerinde doğrusaldır; her geçiş, Jacobi ön koşullu eşlenik gradyanla birkaç kez yeniden ağırlıklandırılan tek bir seyrek en küçük kareler sistemidir (doğrudan ayrıştırma 150 bin bilinmeyende 446 sn, CG 22 sn — aynı cevap).

Ölçüm (örtüşme piksellerinde gradyan büyüklüğü 40'tan fazla ayrışanların oranı): rijit 0.179 → yalnız SIFT 0.137 → iki geçiş 0.089. `warp_camera` yer değiştirmeyi tüketir: kaynağı `(u, v)` yerine `(u − du, v − dv)`'de örnekler; sarmalama şeridi de aynı ızgarayla yeniden render edilir. Optik akış onarımı (`refine.py`) yerinde duruyor, ama artık bir iş yapmıyor.

### Dikiş seçimi

`cv2.detail_GraphCutSeamFinder("COST_COLOR")` — dikişi, iki görüntünün renk olarak en çok anlaştığı yerden geçirir. Yani düz duvarlardan ve gölgelerden; kapı çerçevelerinin ve masa kenarlarının ortasından değil. *Bu, seyircinin dikişi görüp görmemesini belirleyen tek karardır.* Üç ölçülmüş düzeltme:

- **0..255, 0..1 değil.** OpenCV her kenara sabit bir `weight_eps = 1.0` ekler ve uyuşmazlığı renk farkının karesiyle ölçer. 0..1 ölçekli bir görüntüde tam kontrastlı bir uyuşmazlık o epsilona karşı ~3 ağırlığındadır, kesim en kısa yola dejenere olur ve hizalanmamış nesnelerin ortasından dümdüz geçer. 12 px kaymış dokulu bir fikstürde: 0..1'de 300 satırın 150'si dokudan geçiyor, 0..255'te 24'ü.
- **COST_COLOR, COST_COLOR_GRAD değil.** İkincisi renk uyuşmazlığını gradyan büyüklüğüne böler — yani tam olarak *hizalanmamış bir kenarın* büyüttüğü şeye — ve kesimi kaçınması gereken korniş ya da dolap çıtası boyunca yönlendirir. Mesh'le hizalanmış gerçek çekimde: dikiş piksellerinde iki görüntünün uyuşmazlığı 16.4 → 14.8 gri seviye, dikiş üstündeki yapı 42.6 → 36.6, `seam_gradient_energy` 0.93 → 0.74, ve kesim 2.5× hızlı.
- **Sıkı pencereler ve gerçek bütçe.** `footprint_window` eskiden kutbu "kutba yarım dikey görüş açısı kadar yakın" diye tahmin ediyor ve bu 20° erken tetikleniyordu: −40°'ye eğik bir alt satır görüntüsü 8192 px genişliğinde, %12.5'i geçerli bir pencere alıyordu — 5× warp maliyeti, 3.4× harmanlama işi ve dikiş grafiği o kadar büyüyordu ki graph-cut reddedilip dikişler 512 px'te dinamik programlamaya düşüyordu. Kutup artık gerçekten görüntüye yansıtılarak test edilir. Bütçe de ikiye ayrıldı: OpenCV grafı çift çift kurup serbest bıraktığı için **bellek** en büyük tek örtüşmeye, **süre** toplamına bakar.

Yumuşak kareleri dikişten dışlayan maske (`_mask_out_the_soft`) yalnız *tutarlı* bölgeleri teslim eder; benekli dışlamalar sahipliği yüzlerce adaya bölüyordu (ham maskeyle 647 ada, morfolojik açmayla ~100).

> OpenCV 5.0 notu: `find()` maskeleri `cv2.UMat` olarak döndürür. `cv2.UMat.get()` çağrılmadan numpy ile karşılaştırılırsa `TypeError` alınır.

**Dikiş, mimarinin üstünden geçmez.** Graph-cut kesim maliyetini renk farkından hesaplar ve duvarları düz, dokusuz bir odada en ucuz yol tam da en güçlü mimari çizginin üzerinden geçer — korniş, gardırop çıtası, kapı kasası. İki taraf birkaç piksel bile ayrıysa sonuç panoramanın en kötü kusurudur: kapı üstünde beş paralel korniş, köşede birbirinin üstünden geçen iki profil, beyaz bir kapı kenarının kesik kesik çizgiye dönüşmesi.

OpenCV'nin `GraphCutSeamFinder`'ı kendi maliyet fonksiyonuna müdahale ettirmez, o yüzden ceza **görüntüye** yazılır: uzun tutarlı çizgilerin haritası çıkarılır ve kesimin oradan geçmesini pahalı kılacak kadar yapay bir renk farkı iki görüntüye zıt işaretlerle eklenir, kesim bulununca geri alınır. Ceza sıfırken sonuç tek çağrılık graph-cut ile bit-bit aynıdır ve bunu bir test sabitler. Ceza büyüklüğü grinin 22 seviyesi: dikişlerin zaten taşıdığı ~15 seviyelik uyuşmazlığın üstünde, ama OpenCV'nin 10000'lik terminal maliyetinin çok altında — üstüne çıkmak, max-flow'un tek kameranın gördüğü bir pikseli terk etmesini tercih ettirir ve bu daha iyi dikiş değil, yetim piksel üretir. İkinci terim düz kesiğe karşıdır: uzun, eksene paralel bir kesim, aynı renk farkına sahip kıvrımlı bir kesimden pahalıdır, böylece dikiş perdenin kıvrımlarına saklanır, üstünden cetvelle çizilmiş gibi geçmez.

Ölçüm (gerçek üç satırlı çekim, 4096): dikiş gradyan enerjisi 0.725 → 0.696, çizgi düzlüğü 3.58 → 3.06 px, ikilenen çizgi çifti 495 → 479, süre 170 → 175 sn. Uyarı, ölçen ajanın kendi notuyla: `straightness_px` bu boydaki bir değişimi çözemez (yalnızca dither tohumunu değiştirmek onu 3.23 → 3.73 oynatıyor) ve ikilenen çizgi sayımının gürültü genişliği ~21'dir; güvenilir olan dikiş gradyan enerjisi ile gözle görülen sonuçtur.

### Harmanlama

Laplacian piramidi tabanlı çok bantlı harmanlama (`cv2.detail_MultiBandBlender`). Düşük frekanslar geniş bir bantta yumuşatılır (pozlama farkı kaybolur), yüksek frekanslar dar bir bantta kalır (detay bulanıklaşmaz). 8K'da bellek gerçek bir kısıttır; `blend_tiled` yatay şeritlere bölerek aynı sonucu sabit bellekle üretir.

Harmanlayıcıya **sert** maske verilir (2 px tüy), `W/400` = 8K'da 20 px değil. `MultiBandBlender` verilen maskeyi *en ince* bandın ağırlığı olarak kullanır; 20 px'lik bir tüy en yüksek frekansları 32 px'lik bir bölgede çapraz geçirir ve iki görüntünün birkaç pikselden fazla ayrı koyduğu her yapı orada iki kez çizilir. Geniş düşük-frekans geçişini piramit zaten kendisi yapar.

Hayalet onarımı (`repair_ghosted_regions`) kapsama kenarını artık aday saymaz: tuval kapsama dışında tam siyahken kareler en yakın renkle doldurulmuş kenar boşlukları taşır, dolayısıyla yüksek geçiren fark bütün dış sınır boyunca yapı gereği büyüktür (kenarda 11 gri seviye, gerçek 12 px'lik bir paralaks hayaletinde 1.3). Kenar içeride bırakılınca kapatılan bütçenin %96'sı ona gidiyor ve onarımın var olma sebebi olan nesneler hiç seçilmiyordu.

### Paralaks onarımı (stüdyo modu)

Global hizalama en iyi rotasyonu bulur; kalan hata paralakstır ve yereldir.

1. Örtüşme bölgesinde yoğun optik akış (CUDA varsa RAFT, yoksa OpenCV DIS).
2. **Akıl sağlığı kapısı:** medyan akış `flow_max_shift_px`'i aşıyorsa çift tamamen atlanır. Büyük akış, global hizalamanın yanlış olduğu anlamına gelir; warp etmek durumu kötüleştirir.
3. Düzeltme ikiye bölünür: i görüntüsü `+0.5·akış`, j görüntüsü `−0.5·akış` kadar warp edilir — ortada buluşurlar.
4. Düzeltme, dikişte 1 olup örtüşmenin dışına doğru 0'a inen bir ağırlıkla çarpılır ve akış alanı bulanıklaştırılır, böylece düz çizgiler kırılmaz.

### Kutuplar

Tek yatay tur, tavanı ve zemini boş bırakır. Equirect'te kutba yakın inpaint yapmak sürüklenme izleri bırakır, çünkü orada bir piksel neredeyse hiç katı açı kaplamaz.

Doğru yol: kapağı **stereografik** projeksiyona açmak (orada geometri neredeyse konformdur), `cv2.xphoto.inpaint(INPAINT_FSR_BEST)` ile doldurmak ve 6°'lik yumuşak bir bantla geri kompozit etmek.

Üç ölçülmüş düzeltme: kapak **kapsamaya göre normalize** örneklenir (kapsama kenarını kesen düz bir bilineer örnek dışarıdaki siyahı halkaya karıştırıyor ve dolgu sınırı boyunca noktalı koyu bir çizgi bırakıyordu; sütunların %19'unda 20 seviyeden fazla çukur); halka eğiliminin ekstrapolasyonu **boşluğa göre sınırlanır** ve kelepçeye çarpan bir eğim güvenden düşer (referans çekimde +4/derecelik kelepçeli eğim 36° taşınıp halkanın 240'ını kapağın tamamında düz 255'e — "beyaz leke"ye — itiyordu); dolgu, halkadan içeri doğru kendi azimut ortalama profiline çekilir (Telea halkanın açık/koyu yaylarını kutba kadar bulut gibi taşıyordu).

---

**Kutup dolgusu gerçek pikselin üstüne yazmaz.** Üç satırlı el çekimi kürenin ~%85'ine ulaşır; tavan ve zemin kapakları sentezlenir. Eski dolgu düz tavanda inandırıcı, başka her yerde kötüydü: odanın beş kaseli tavan lambası — panoramanın başka yerindeki aynada net görünen — iki yırtık yarım kaseye iniyordu, kapak sınırı çokgen bir kubbe kenarı gibi duruyordu (yarıçap boyunca ortalama gri 192'ye düşüp 198-209'a sıçrıyor, yerel standart sapma 10 pikselde 6.3'ten 18'e fırlıyordu) ve nadirde halının deseni logoya yaklaşırken ışınsal bulanıklığa dönüşüyordu.

Üç değişiklik: dolgu, kapak sınırında hâlâ gerçek kapsama ve yapı bulunan yerlerde gerçek pikselleri **uzatır**, üstlerine yazmaz; sınır çokgen maske yerine derecelerle ölçülen yumuşak bir yarıçap geçişidir ve dolgunun hem yerel ortalaması hem **taneciği** hemen dışındaki gerçek halkaya uydurulur (yoksa parlaklık uymasa bile doku uymaz ve sınır yine görünür); kapak çözünürlüğü tuvalden türetilir, çünkü kutup deliği dışarıdan içeriye bir **ekstrapolasyondur** ve kenarda esirgenen her çözünürlük, birleşeceği halkaya karşı bir uyumsuzluktur.

---

## 8. Çıktı ve oynatıcı

Dört seviye üretilir: 512 / 2048 / 4096 / 8192. Oynatıcı önce 512'yi yükler (anında bir şey görünür), sonra cihazın kaldırabileceği en yüksek seviyeye **çapraz geçişle** yükselir.

Cihaz sınırı ciddiye alınır: 8192×4096 RGBA + mipmap ≈ 180 MB VRAM. Orta segment bir telefonda bu sekmeyi çökertir. `MAX_TEXTURE_SIZE`, `deviceMemory` ve işaretçi tipine bakılarak tavan belirlenir.

**Renk doğruluğu:** `MeshBasicMaterial` + `BackSide`, ışık yok, `NoToneMapping`, sRGB. Panorama bir fotoğraftır; render edilmez, yeniden üretilir. ACES gibi bir ton eşlemesi burada görüntüyü gözle görülür şekilde soldurur.

**Görüş açısı yataya kelepçelenir.** `fov` three.js'in *dikey* açısıdır ve en-boy oranından bağımsız uygulanıyordu: 75°'lik varsayılan 16:9 bir masaüstünde yatayda 107°, ultra geniş ekranda azamide 140° ediyor ve kare kenarlarında 2.9–5.5× rektilineer gerilme bırakıyordu — kusursuz bir panoramada bile korniş kavis alır, yakın nesneler yayılır. Türetilen yatay açı bir tavanı geçmeyecek şekilde dikey değer düşürülür — ama bu tavan **ultra geniş ekran için bir emniyet payıdır, bir zoom sınırı değildir**, ve fark önemli: 100°'de tam olarak zoom sınırıydı. `FOV_MAX` 100° dikey alandır, bu da 16:9 bir pencerede yatayda 129° eder; yatayı 100°'de kesmek dikeyi 68°'de tutuyordu, yani zoom-out yolunun son üçte biri sayıyı değiştirip ekranda hiçbir şeyi değiştirmiyordu. Uzaklaştırma sessizce duruyor, hissedilecek bir yol sonu da olmuyordu. 130° her ekranda (16:9'a kadar) dikey azamiyi serbest bırakır — zoom aralığı yeniden bütündür — ve 21:9 bir monitörün alacağı 140°'lik gerilmeyi hâlâ budar. `lookControls.test.ts` bunu her yaygın en-boy oranı için sabitler: zoom-out sınırına gerçekten ulaşılabilmelidir. **Pitch kapsama bandına kelepçelenir:** çekimin ulaşmadığı tavan ve zemin sentezlenmiş kapak ve marka işaretidir; oynatıcı sahnenin `coverage.pitch_min/max_deg`'ini birkaç derece payla sınır alır, kutup tam kapsanmışsa serbest bırakır.

---

**Bir kapı iki kez sorulmaz.** Yakalama uygulamasında iki kez aynı hata yapılmıştı: uyarlanmış bir kararın yanına ham sabit konması, ki bu uyarlamayı sessizce iptal eder.

`coverage.gatesForLens` kapsama eşiğini merceğin ulaşabileceğine göre gevşetir — sabitler 9:16 karede ölçüldü, orada üç satır +55/-71'e ulaşır ve kusursuz bir süpürme 0.886 alır; 4:3 kare 14° daha kısadır ve orada **kusursuz** bir süpürme ancak ~0.80 alır. `plan.verdict` ise `blockedBy`'ın yanında ham 0.855'i de sınıyordu. Sonuç: 4:3 bir telefonda her satırın her istasyonu dolu, hiç boşluk yok, ufuk %100 — ve yine "tekrar çek". Gerçek çekimde ölçüldü (vivo Y22s, 1080×1440): satırlar 1.00/1.00/1.00, boşluk 0, ufuk 1.00 ve 0.9997, pitch erişimi +56/-75, kapsama 0.8314 → reddedildi. Kullanıcının yapabileceği hiçbir şey kalmamıştı.

Aynısı yürüme kapısında: `walkingGrade` ancak hız tahmini `stopSustainMs` (1500 ms) boyunca eşiğin üstünde kalırsa kırmızıya döner — mandal tam da bu yüzden var, çünkü bir itiş ya da tutuşu düzeltme tahmini birkaç yüz milisaniye eşiğin üstüne çıkarır ve yüksek geçiren filtre onu geri alır. `verdict` bunun yanına ham `maxDriftMPerS` zirvesini de koyuyordu. Aynı çekimde: mandallanan %2.5, medyan sürüklenme 0.043 m/s, tek zirve 0.4028 → "yürüyor" damgası. Motor aynı çekimi 0.76°'lik döngü kalıntısıyla çözdü, ki bu yürünmüş bir çekimin görüntüsü değildir.

Her iki durumda da düzeltme aynı: kararı veren mekanizma tektir ve ham eşik onun yanında ikinci bir oy kullanamaz. `plan.test.ts` ikisini de gerçek sayılarla sabitler.

---

## 9. Ölçüm

`engine/biosphere/synth/` analitik bir oda ışın izleyicisidir. Prosedürel bir odayı hem equirect panorama olarak (yer gerçeği) hem de sanal bir telefon kamerasından kare kare render edebilir. Paralaks bedavaya ve tam doğrulukla gelir: kamera merkezi kürenin merkezinde olmadığı için oradan yeniden ışın izlemek fiziksel olarak doğru cevaptır.

Karşılaştırma iki tuzaktan kaçınır:

1. **Gauge belirsizliği.** Dikilmiş panorama global bir dönüşe kadar tanımlıdır. Ham PSNR anlamsızdır; önce `align_panoramas` ile hizalanır.
2. **Kutup ağırlığı.** Equirect tuvalde ağırlıksız ortalama, kürenin neredeyse hiçbir yerini kaplamayan kutupları baskın hale getirir. Tüm ortalamalar `sphere_solid_angle_weights` ile ağırlıklandırılır.

Kabul eşikleri `tests/test_pipeline_quality.py` içinde iki kademelidir: **nodal** (paralakssız, algoritmayı izole eder) ve **realistic** (28 cm kol yarıçapı, ticari olarak anlamlı olan).

---

## 10. Servis mimarisi

- **İş kuyruğu:** süreç içi `ThreadPoolExecutor`, SQLite iş kaydı. Celery'ye geçiş tek bir sınıfın değişmesi demek. MVP için ayrı bir broker'ı işletmeye değmez.
- **İlerleme:** pipeline aşama bazlı raporlar; `StageWeightedProgress` bunları tek bir 0..1 izine ölçekler ve **geri gitmemesini** garanti eder. Aşama ağırlıkları gerçek ölçümlerden gelir — %12'de doksan saniye duran bir çubuk, kullanıcıya "donmuş" der ve iptal ettirir.
- **CORS:** oynatıcı üçüncü taraf emlak portallarına `<iframe>` ile gömülür. Genel okuma rotaları ve `/media` her origin'e açıktır ve `Cross-Origin-Resource-Policy: cross-origin` gönderir — bu başlık olmadan tarayıcı dokuyu WebGL bağlamında kullanmayı reddeder ve oynatıcı sessizce boş kalır.
- **Ölçüm (`/ai/measure`) gerçek geometridir, taklit değil.** Kamera yüksekliği bilinen bir panoramada, zemine işaretlenen bir noktanın ışını `y = kamera_yüksekliği` düzlemini keser; bu gerçek bir 3B noktadır. Ufkun üstündeki noktalar zemini kesmez ve `null` döner — uydurulmuş bir sayı dönmez.
- **Çekim bağlantısı (QR) giriş istemez, ama hiçbir şeye de açılmaz.** `/cek` QR kodunu çizen cihazda değil, telefonda açılır; telefonun önüne giriş formu çıkarmak, müşterinin evinde kamera açıkken şifre yazdırmak demekti. Ev sayfasındaki QR artık `k` taşır: o eve bağlı, 12 saat yaşayan bir `CaptureGrant` (veritabanında yalnızca özeti durur). `deps.current_user` oturum bulamazsa bu bağlantıya bakar ve onu evin sahibi olarak kabul eder — ama yalnızca `_CAPTURE_ROUTES` listesindeki rotalarda (evi okumak, oda açmak, yüklemeyi parça parça göndermek ve birleştirmek, başlattığı işi izlemek) ve yolun içindeki her kimlik o eve aitse. Liste bir **izin listesidir**: yarın eklenen bir uç nokta, biri onu bilerek listeye koymadıkça QR'dan erişilemez. Evin sahibi değişirse ya da `DELETE /properties/{id}/capture-links` çağrılırsa bütün kodlar ölür. Telefon anahtarı adres çubuğundan alıp sekmenin `sessionStorage`'ına taşır (adres ekran görüntüsüne, sohbetlere gider); motor bağlantıyı reddederse telefon giriş sayfasına değil "yeni kodu okutun" ekranına düşer. Yeni bağlantıyı yalnızca bir oturum üretir: bağlantıyla açılmış bir sekme kendi bağlantısını aynen paylaşır, çünkü onunla üretmeye kalkmak 401 alır ve istemci elindeki bağlantıyı da atardı.

---

## 11. Sanal dekorasyon

Aynı oda, beş farklı dekorasyon stilinde. `engine/biosphere/ai/restyle.py` (fotoğraf) ve
`restyle_pano.py` (küre).

### Neden boşaltmanın tam tersi

`sd_inpaint.py`, difüzyon inpainter'ını oda boşaltma için ölçtü ve **reddetti**: %18'lik bir
maskede yatak şeklindeki deliğe yepyeni bir yatak çizdi, %74'te başkasının evini çizdi, ve
guidance'ı 6'dan 15'e taramak da mobilyayı negatif prompt'ta ismen saymak da bunu değiştirmedi.
Bulgu şuydu: üretken bir inpainter *"bu deliğe ne ait"* sorusunu cevaplar, ondan bir **yokluk**
istenemez.

Dekorasyon tam olarak o soruyu sorar. O yüzden orada kusur olan davranış burada mekanizmanın
kendisidir ve ölçülmüş itiraz bu tarafa geçmez.

### Odayı yerinde tutan üç şey

Bu özelliğin zor kelimesi "**aynı** oda". Stil prompt'u güçlü bir talimattır, fotoğraf zayıf bir
talimattır; rustik bir salon isteyen model size memnuniyetle **başka bir yerde** güzel bir rustik
salon verir. Üç arıza sırayla ölçüldü:

1. **Düz min–max normalize edilmiş derinlik → oda yer değiştirdi.** Suç ControlNet'te değildi,
   haritadaydı: kadrajdaki açık kapı (çok uzak) ve merceğe 30 cm mesafedeki yatak köşesi (çok
   yakın) tüm aralığı yiyordu, odayı tanımlayan bütün yüzeyler yirmi gri seviyeye sıkışıyordu.
   Yüzdelik kırpma (`_DEPTH_CLIP_LOW/HIGH`) kapının yerini beş stilin üçünde sabitledi.

2. **Doğru derinlikle bile güçlü prompt altında sürüklendi.** Derinlik *yüzeylerin nerede
   olduğunu* kısıtlar; odayı o oda yapan düz çizgiler — tavan birleşimi, kapı kasası, duvarın
   zeminle kesiştiği hat — hakkında hiçbir şey söylemez, çünkü onlar kenar, mesafe değil. İkinci
   koşullayıcı bu yüzden var: **sadece mimariye maskelenmiş Canny.** Kenarlar ham fotoğrafta
   hesaplanıp segmenter'ın yapı dediği piksellere kırpılır — kapı kasası kilitlenir, **eski
   kanepenin silueti kilitlenmez.** Onu da kilitlemek özelliğin tersi olurdu: yeni mobilya
   eskisinin dış hattı üzerine çizilirdi.

3. **İki koşullayıcıyla da sürüklendi — çünkü fotoğraf başlangıç latent'inde hiç yoktu.**
   `strength` pipeline'ın varsayılanı olan 1.0'daydı, yani maskeli bölge saf gürültüden
   başlıyordu: iki ControlNet odanın söyleyebildiği her şeydi ve stil prompt'u onları oyladı.
   Ölçülen tarama:

   | strength | sonuç |
   |---|---|
   | 1.00 | başka bir daire. Rustik: şömineli taş kulübe. Modern: yerleşim değişmiş. |
   | 0.85 | oda duruyor, en güçlü stil. Rustik duvarı ahşap kaplamaya başlıyor — bu bir *yüzey*, izin verilen tarafta, ama sınırı. |
   | 0.70 | oda duruyor, stil okunuyor, orijinalin dağınıklığı hafif görünüyor. Güvenli uç. |
   | 0.55 | eski oda, renk filtresi geçmiş. Laptop ve çanta hâlâ okunuyor. |

   Bu tarama **duvarın da düzenlenebilir olduğu** zaman koşuldu; sayının temkinli olmasının
   sebebi bu — her strength puanı, duvarı oynatma iznine bir puandı. `replace` modunda duvar
   zaten düzenlenemez, dolayısıyla o izin yok ve temkin bedava değil: `stage` modunda **0.80**,
   `replace` modunda **0.95**. Düşük strength'in satın aldığı şey — yeni yatağın eskisine
   benzemesi — `replace` modunda tam olarak istenmeyen şeydir.

### Dördüncü arıza: oda değil, odanın İÇİ değişmeli

Yukarıdaki üç düzeltmeden sonra oda yerinde duruyordu ama maske hâlâ "korunan mimari dışındaki
her şey"di — yani duvar ve zemin modelin çizebileceği alandı. Gerçek bir 360 turda bunun bedeli
görüldü: **model perdeli bir pencerenin önüne dolap çizdi.** Bu bir ayar hatası değil, maskenin
istediği şeydi; duvar serbestse duvar dolap için gayet uygun bir yerdir.

Doğru soru şuydu: *yatak varsa modern yatak olsun.* Yani odanın kabuğu hiç değişmesin, içinde
duran eşya aynı türden yenisiyle değişsin. Modül artık modu **odadan okuyor**, çağırandan değil:

`replace` Segmenter mobilya buldu. Düzenlenebilir olan **yalnızca o pikseller**; duvar, zemin,
          tavan, pencere, kapı, perde ve tesisat armatürleri korunuyor. Prompt'taki isimler de
          odadan geliyor — segmenter "bu odada yatak ve gardırop var" diyebiliyor, prompt da
          onu söylüyor.
`stage`   Oda boş (`_FURNISHED_FRACTION` altında). Değiştirilecek bir şey yok, o yüzden mobilya
          **zemin ve üstündeki bant** içine uyduruluyor (`_furnishable_volume`) — bütün duvara
          değil, çünkü düzenlenebilir bir üst duvar, modelin mülkte olmayan bir pencere çizdiği
          yerdir.

Bu ayrım `Style` sınıfını da böldü: stil artık yalnızca **malzeme ve renk paletine** sahip,
mobilya isimlerine değil. "Düşük gri kanepe, ceviz konsol" yazan bir stil, yatak odasında
yatağın yerine kanepe çizdirir.

Ölçüldü: aynı gerçek yatak odası penceresinde düzenlenebilir alan **%100 → %32**, perdeler
dokunulmadan kaldı, ve pencere başına süre 7.6 s → 5.9 s'e indi. 360 turda küre başına 142 s →
94 s, ve kürenin **%68'i artık hâlâ gerçek fotoğraf**.

### Neyin değişmesine izin var, neyin yok

Bu görsel alıcının önüne konuyor, o yüzden çizgi modelin iyi olduğu yerden değil, **yanlış
beyanın başladığı** yerden geçiyor.

*Değişebilir* — mobilya, tekstil, halı, yere duran aydınlatma, duvar rengi ve yüzeyi, zemin
kaplaması. Hepsi alıcının taşındıktan sonra kendi değiştirebileceği şeyler.

*Değişemez* — pencere boşlukları ve **pencereden görünen her şey**, kapılar, kolonlar, merdiven,
tesisata bağlı armatürler (lavabo, klozet, küvet, duş, radyatör, ocak, tezgâh). Bunlar mülke ait
gerçekler. Özellikle pencere manzarası insanların para ödediği bir şeydir; uydurulmuş bir manzara
stil değil, yalandır.

Koruma **kompozitle** uygulanır, modelden rica edilerek değil: o pikseller geçişten sonra
orijinal fotoğraftan geri gelir. `feathered_alpha` yumuşamayı yalnızca düzenlenebilir tarafa
koyar, böylece maskenin içinde ağırlık tam olarak 1.0'dır ve pencere **bit bit** aynı penceredir
— `test_restyle.py` bunu böyle doğruluyor.

Perdeler ve stor **varsayılan olarak korunuyor**, ve bu bir tercih değil bir düzeltme: ilk
sürüm onları "dekorasyon tercihi" sayarak serbest bırakmıştı ve ilk gerçek 360 turda pencerenin
önüne dolap çizildi. Bir perde, bir pencerenin olduğu yerdir; üstüne çizilen her şey camın
üstüne çizilmiştir. `window_dressing=False`, perdesi gerçekten değişmesi gereken ve penceresi
zaten görünen bir oda için geri çıkış yolu.

### Küre: asıl zorluk projeksiyon değil, anlaşma

`room_fill` delik doldurur — her pencerenin çoğu zaten fotoğraftır, komşular doğal olarak
anlaşır. Dekorasyonun böyle bir çıpası yok: her pencereden mobilya *uydurması* istenir ve "bir
İskandinav salonu"nun on altı bağımsız yorumu, sekiz dikişte buluşan on altı farklı dairedir.

Üç şey anlaştırır: her pencere panoramanın **o anki halinden** render edilir (bağlam, komşunun
cevabıdır); komşunun döşediği yer bu pencerede **korunur, yeniden çizilmez** (`analyse_room`'un
`extra_protect` parametresi bunun için var); ve halka sırayla dolaşılır, ilk pencere tonu belirler.

Kutuplar döşenmez, korunur: iki sıra pencere ±75°'ye ulaşır, ötesi tepedeki tavan ve ayak
altındaki zemindir — equirect'te neredeyse hiç katı açı kaplamayan, elde çekimde çoğu zaman
çekenin kendi ayakları olan bölge.

**Çözünürlük burada bir kalite ayarı değil, çalışıp çalışmama farkı.** Aynı oda, aynı stil:
2048 tuval / 512 pencere 115 sn sürdü ve duvarların yarısı beyaz lapa olarak geldi; 4096 tuval /
1024 pencere 122 sn sürdü ve bir oda verdi — dikişin iki yanında aynı kaktüs. Yedi saniye fark,
çünkü difüzyon geçişi her iki durumda da kendi latent boyutunda koşuyor; değişen sadece iki
yanındaki projeksiyondu.

### Maliyet ve borç

Fotoğraf: geçiş başına ~6.5 sn, beş stil ~35 sn. Küre: stil başına on altı geçiş, ~2 dakika.
Her ikisi de diskte önbelleklenir; ikinci istek milisaniyedir.

**İkisi de birer iş (job) olmalı, henüz değiller.** `/ai/empty-room` ile aynı borç ve aynı sebep:
`core.jobs` kuyruğu bir dikiş etrafında şekillenmiş — bir sahne kimliği ve dikişin kendi aşama
listesi ile — ve onu genişletmek bir şema değişikliği. O güne kadar ikisi de uzun bir istek
zaman aşımı altında koşuyor, ve bunu saklamak yerine yazmak doğrusu.

---

## 12. Oda: 360, fotoğraf, ya da ikisi

`Scene` başlangıçta "bir 360 bakış noktası"ydı. Ürünün gerçek birimi ise **oda**: bir ilanda
salonun panoraması da vardır, dört tane normal fotoğrafı da, ve ikisi aynı odaya aittir.

Bu yüzden bir oda artık şunlardan herhangi biri olabilir: yalnızca 360, yalnızca fotoğraf, ya da
ikisi birden. Çekimi olmayan bir oda `capture_kind = "none"` taşır, işi ve panoraması yoktur,
ve var olduğu anda `ready`'dir — beklenecek bir şey yoktur.

**`ScenePhoto` kendi tablosu, JSON kolonu değil.** Sahnedeki diğer zengin alt nesneler
(`panorama`, `coverage`, `metrics`) worker tarafından bir kez yazılıp bütün belge olarak
okunuyor; JSON kolonu tam olarak bunun için iyidir. Fotoğraflar öyle değil: birkaçı birden
ekleniyor, tek tek siliniyor, sıralanıyor. SQLAlchemy düz bir JSON kolonunun içindeki mutasyonu
izlemez, dolayısıyla her işlem bütün listenin oku-değiştir-ata döngüsü olurdu — iki sekme açık
olduğu ilk anda kaybolan bir güncelleme ile birlikte.

Baytlar sahnenin kendi dizininde duruyor, böylece bir turu silmek fotoğrafları da aynı
`delete_tree` ile götürüyor.

**Yükleme kısmi başarıya izin veriyor.** Dört sağlam dosya ve bir bozuk dosya geldiğinde beşini
birden reddetmek, birini cezalandırmak için kullanıcıya dört yükleme ödetirdi. Sunucu okunabilen
dosyaları saklıyor ve yanıtta hangilerinin geldiğini söylüyor; **hiçbiri** görsel değilse 415.

Fotoğraflar public yanıtta da var, çünkü onları kimin için yüklediğimiz belli: turu gömen ilan
portalı. Bir Chrome eklentisinin ya da bir dışa aktarmanın tek soru sorması yeter — çocuk odası
için elimizde ne var?

### Odanın fotoğraflarında da aynı iki araç

Oynatıcı bir 360 üzerinde iki şey sunuyor: dekore et, ya da mobilyaları kaldır. Odanın normal
fotoğrafları da aynı ikisini hak ediyor, ve arkalarında **yeni bir algoritma yok**:

- **Dekorasyon** — `ai.restyle` zaten herhangi bir fotoğrafta çalışıyor.
- **Boşaltma** — `ai.empty_photo`, `ai.empty_room`'un perspektif-tuval kardeşi ve çok daha
  küçük, çünkü o modülün zor olan her yanı *tuval bir küre olduğu için* zordu: equirect'te
  süpürgelik bir sinüs, dikiş sütunu sarmalanıyor, kutuplar katı açı kaplamıyor. Fotoğraf
  bunların hiçbiri değil; zaten her modelin üzerinde eğitildiği şey.

**İkisi de aynı maskeyi kullanıyor** — `restyle.read_room`. Bu bir kolaylık değil, tasarım:
dekorasyon o pikselleri yeniden çiziyor, boşaltma onları siliyor, ve tek bir "mobilya nedir"
tanımı iki özelliğin bir perde konusunda ayrı düşmesini engelliyor. Aksi halde aynı kullanıcı
birinde korunan perdeyi diğerinde silinmiş bulurdu.

Dolgunun gerçekte ne ürettiği, ölçülmüş
--------------------------------------
Gerçek bir dikilmiş yatak odası panoramasının iki 90 derecelik kesitinde:

| delik | sonuç |
|---|---|
| %21.6 | yatak ve sandalye gitti, perdeler ve duvarlar sağlam; yatağın olduğu zemin ahşap yerine gri bir yıkama olarak geliyor |
| %33.4 | komodinin bulunduğu köşede koyu bir leke |

Sebep olarak **elenenler**: LaMa'nın çalışma çözünürlüğü (384, 512, 768 ve 1024 aynı lekeyi
veriyor), maskeli bölgenin çağrıdan önce sıfırlanıp sıfırlanmadığı (izlenmiş model bunu kendi
yapıyor), ve deliği bağlantılı bileşenlere bölüp en küçüğünden doldurmak — `room_fill`'in kürede
kullandığı desen, ki burada yalnızca üç bileşen buldu ve hiçbir şeyi değiştirmedi.

Yani bu, maskedeki bir hata değil (maske piksel piksel doğru, `scripts/bench_empty_photo.py`),
dolgunun bu malzemedeki tavanı. Aynı zamanda `/ai/empty-room`'un kürede zaten sunduğu tavan,
çünkü ikisi de aynı inpainter'da bitiyor.

**Dürüst araç bu yüzden bir eşik değil, karşılaştırmanın kendisi.** Sonuç asla tek başına
gösterilmiyor: orijinalin üstüne sürülmüş halde, `/dekor` sayfasının kullandığı kaydırıcıyla
çıkıyor, ve yanında kaç yüzdenin yeniden oluşturulduğu yazıyor. Kolu geri çekip kendi kararını
veren kullanıcı bunu yayınlayıp yayınlamayacağını bilir; sadece çıktıyı görene bilemez.
`MAX_HOLE_FRACTION` yalnızca en baştan umutsuz olan vakayı yakalıyor.

Türetilmiş sürümler orijinalin `mtime`'ına göre önbellekleniyor, kimliğine göre değil. Bir
fotoğraf pratikte değişmez — kimlik bir satır, dosya bir kez yazılıyor — ama "pratikte" tam
olarak bayat türevlerin sunulmaya başladığı yerdir, ve kontrolün maliyeti bir `stat`.

### Boşaltma, odanın kendi yüzeyleriyle (Ekim 2026)

Panoramada mobilya kaldırma artık LaMa'ya bırakılmıyor: `ai/room_planes.py` tavan ve zemin
çizgisinden bir oda kutusu oturtur ve her yüzeyi (duvar, zemin, tavan) **o yüzeyin kendi görünen
boyasından, süpürgeliğinden, perdesinden ve zemin deseninden** yeniden kurar; LaMa yalnızca
artakalan ince şeritleri doldurur. Maske de genişledi: segmentörün "oda" dediği ama oda kutusunun
dışında kalan şeyler, aykırı renkler ve temas gölgeleri eklenir; kapılar asla eklenmez. Üç gerçek
yatak odasında, doldurulan alanın kendi yüzeyinden 15 L*'tan koyu kalan payı 0,43/0,71/0,47 →
0,08/0,11/0,09; yeniden segmentlenen çıktıda "hâlâ mobilya" payı %4,4 → %0,6 ve %11,2 → %0,2.
Bilinen kusurlar: zemin desensiz ve hafif gri çıkar, segmentörün kaçırdığı küçük eşyalar kalır,
dağınık odada kutu oturtma kırılgandır. Düz fotoğraflarda düzlem dolgusu yok — fotoğrafın kendi
çizgilerinden kamera kalibrasyonu 3–8° eğim ve %60'a varan odak hatası verdi; fotoğraflarda maske
tamamlama + LaMa sürüyor.

## 13. Zemin halkaları: turda yürümek

Bir turun odaları vardı ama turun *içinde* odadan odaya geçmenin yolu yoktu; geçiş, oynatıcının
dışındaki bir listeden seçmekle oluyordu. Bu, evi gezmeyi bir menüden şıklamaya çevirir. Halkalar
bunu tersine çeviriyor: kapı eşiğinin önündeki zemine bir işaret konuyor, tıklayan oraya geçiyor.

### Halka neden bir mesh, diğer bütün katmanlar HTML iken

`HotspotLayer` etiketleri DOM düğümü olarak çiziyor ve gerekçeleri sağlam: metin net kalır,
odaklanabilir, ekran okuyucu okur, ve etiket nerede olursa olsun aynı boydadır. Bu gerekçelerin
hepsi halkada çöküyor, çünkü **halka bir etiket değil, bir yer.**

Bir yerin zemine yatması ve perspektife uyması gerekir: yakında geniş bir elips, uzak duvarın
dibinde ince bir çizgi, yaklaştıkça büyüyen. Ekran uzayında düz bir daire olarak çizildiğinde
odanın önünde duran bir düğme gibi okunuyor ve kullanıcıya oraya yürüyebileceğinin *söylenmesi*
gerekiyor. Zemin düzlemine çizildiğinde bunu resmin kendisi söylüyor. Turun gezilmiş değil
kullanılmış hissettirmesiyle arasındaki fark bu, ve iki üçgene değer.

### Geometri motorun kendi geometrisi

Halka, hotspot'un ışınının zeminle kesiştiği yere oturuyor — `ai/measure.py`'nin tek görüntülü
ölçüm için hesapladığı kesişmenin ta kendisi, oynatıcının çerçevesinde:

    d      = (cos p cos y, sin p, cos p sin y)      hotspot'un yönü, +Y yukarı
    zemin  = y = -EYE_HEIGHT düzlemi
    nokta  = d * (-EYE_HEIGHT / d.y),   yalnızca d.y < 0 iken var

`EYE_HEIGHT = 1.6 m` serbest bir sayı değil: yönü mesafeye çeviren tek şey o, dolayısıyla
halkanın ne kadar uzağa düşeceğini ve ne büyüklükte görüneceğini o belirliyor.

Ufkun üstündeki bir portal **kırpılmıyor, reddediliyor.** Kırpmak cazip hatadır: yanlış konmuş
bir halkayı sessizce makul bir yere taşır, ve koyan kişi duvara koyduğunu hiç öğrenmez.

`MAX_DISTANCE = 14 m` ise ufka çok yakın bir portalin onlarca metre öteye — duvarın arkasına, ve
çizildiğinde piksel altına — düşmesini engelliyor. Bunun doğrudan sonucu, halkanın
konabileceği **en sığ açının ufkun 6.5 derece altı** olması; `atan(1.6 / 14)`. İki sabitten
türeyen bu sayı testte açıkça yazılı, ki sabitlerden birini değiştirmek bilinçli bir iş olsun.

Bu aritmetik `portalGeometry.ts`'de, çizim `PortalRings.tsx`'de duruyor. Ayrım keyfi değil:
yanlış yere düşmüş bir halka ancak odanın içinde durarak görülür, onu oraya koyan kural ise
masa başında ölçülebilir — ve React ile r3f'ten arınmış olduğu için düz Node ortamında koşuyor.

### Tıklama raycast'ten değil, dokunma kararından geliyor

r3f bu mesh'leri seve seve raycast eder — ve kullanıcının zamanının çoğunu geçirdiği jest olan
**bakış sürüklemesini** bitiren `pointerup`'ta da ateşler. Bakış kontrolleri dokunmayı
sürüklemeden zaten ayırıyor, üstelik eşikleri gerçek telefonlarda ayarlanmış; bu yüzden
etkinleştirme o kararı yeniden kullanıyor ve dokunmanın yönünü her portalin kendi yönüyle
eşleştiriyor (`hitTest`). Tolerans açısal: "şuna nişan aldım" demenin karşılığı bu, ve uzaktaki
halka ekranda daha küçük bir hedef oluyor — doğru şekilde, çünkü uzakta.

Hover ise r3f'ten geliyor; hover sürüklemeyle karıştırılamaz.

### Tıklayınca gerçekten oraya yürünüyor

Halka bir *yer* ise, oraya götüren tıklama da bir *yolculuk* olmak zorunda; yoksa yer olduğu
iddiası boşa düşer. Oynatıcının zaten bir kapı geçişi vardı — karart, merceği kıs, değiştir,
aç — ve listeden seçilen bir oda için doğru olan da odur: resimde nereye gittiğinizi söyleyen
hiçbir şey yoktur. Halka için yanlıştır. Zeminde bir noktaya nişan aldınız; kamera **o noktaya**
doğru yola çıkmalı ve oda yolda etrafınızda büyümeli.

**Merkeze çivilenmiş bir kamera nasıl yürür?** Panorama, kamerası tam merkezinde duran bir küre;
görünüşte gidilecek bir yer yok, her piksel eşit uzaklıkta. Ama küre **sonlu**, ve bu yetiyor:
kamerayı kapıya doğru merkezden kaydırdığınızda karşı duvar yaklaşıp büyüyor, yanınızdaki
duvarlar da karenin kenarlarından süzülüp geçiyor. Etkinin tamamı o süzülme. Dolly çekiminin
kullandığı hilenin aynısı ve burada da aynı sebeple işliyor: göz, değişen perspektifi kendi
hareketi olarak okuyor.

**Ayarlanacak şey mesafe değil, mesafenin ne kadarının görüldüğü.** İlk sürüm bunu kaçırdığı için
"gidiyor" gibi okunmuyordu ve ölçünce sebep tekti: perde yolun %55'inde kapanmaya başlıyordu,
yani kullanıcının gördüğü hareket **418 ms** ve kapının büyümesi **1.50 kat**. Yarım saniyenin
altında, bir buçuk kat: göz bunu yürüyüş değil, ani bir zoom ve ardından kesme diye kaydeder.
Yolculuk 1150 ms'ye, perde %72'ye ve dolly yarıçapın 0.72'sine çekilince görünen kısım **828 ms**
ve **2.39 kat** oluyor — 3 metre ötedeki bir kapıya doğru atılan bir adım değil, yürünen bir
mesafe.

Geometrik olarak doğru değil — gerçek bir adım kapı kasasının ardındakini açığa çıkarırdı, bu
çıkaramaz, çünkü fotoğrafta o bilgi yok. Doğru olması da gerekmiyor, yalnızca **kısa** olması:
bozulmanın fark edilebileceği ana gelindiğinde perde çoktan siyah.

Üç ayrıntı kararı taşıyor:

- **Yön kesinlikle yatay.** Halka zeminde olduğu için tıkladığınız yön aşağıyı gösteriyor; o
  yönde ilerlemek kamerayı karolara sürer ve yürümek değil düşmek gibi görünür. Kullanıcının
  "oraya git" derken kastettiği halkanın *kerterizi*, eğimi değil — bu yüzden yalnızca yaw
  kalıyor.
- **Bakış tam düzleşmiyor.** Kapıya dönerken göz zeminden kaldırılıyor ama 7 derece aşağıda
  bırakılıyor; sıfırlamak kameranın dikleşip tıklanan şeyle bağını koparması gibi okunuyor.
  Dönüş yürüyüşle örtüşüyor, "dön, dur, sonra git" değil tek bir hareket.
- **Yumuşatma smoothstep, ve perde ona göre ayarlı.** İçgüdüsel seçim kübik ease-in olurdu,
  çünkü yola çıkmak hızlanmaktır. Burada yanlış: ease-in yolun yarısında mesafenin sekizde
  birini almış olur, ve yolun yarısı kullanıcının gördüğü kısmın çoğudur — yani uzun bir
  tereddüt, ardından karartma. Smoothstep zamanın %72'sinde mesafenin %81'ini alıyor; perde tam
  orada kapanmaya başlıyor. Sondaki yavaşlama siyahın arkasında harcanıyor ve hiçbir şeye mal
  olmuyor.
- **Mercek hiç kıpırdamıyor.** İlk sürüm yolculuk boyunca görüş açısını 7 derece açıyordu — oyun
  motorlarından gelen, "geniş mercek hızı satar" refleksi. Sınırlı bir dolly ile bu refleks
  yanlış, üstelik ölçülebilir biçimde: 75 dereceyi 82'ye açmak ekrandaki ölçeği 0.88 ile
  çarpıyor, yani dolly'nin yeni kazandığı yaklaşmanın **sekizde birini geri veriyor**. Zaten
  dolly-zoom'un tanımı bu: etraf değişirken özneyi aynı boyda tutmak — kendisine doğru
  yürüdüğünüz bir kapının yapması gereken şeyin tam tersi. Kısmak ise büyümeyi artırırdı ama
  etkinin öbür yarısını, yani duvarların kenarlardan süzülmesini keserdi; asıl "ben hareket
  ediyorum" duygusunu veren o. Mercek yerinde duruyor, ki yürümek de zaten odur.
- **Halkalar yola çıkarken çekiliyor.** Halka metreyle yerleştiriliyor (kapı için 3.1 m),
  yolculuk ise kürenin yarıçapının bir kesri; kürenin metrik bir ölçeği yok. İki çerçeve
  uzlaştırılamıyor — 3 metrelik bir dolly 100 birimlik küreyi %3 oynatır, yani görünmez — bu
  yüzden kamera halkayı yolculuğun yaklaşık **sekizde birinde** geçiyor. Olduğu gibi bırakılınca
  halka bütün kareyi kaplayıp bir anda yok oluyor, ki bu hareket değil hata gibi okunuyor. Hızla
  söndürülünce "işaretin üstüne basıp geçmek" gibi okunuyor.

Kamera konumunun **tek bir yazarı** var: `useLookControls` dönüşü yönetip konuma hiç dokunmuyor,
`PortalTransit` konumu yönetip dönüşe hiç dokunmuyor. Yolculuk yokken kamerayı merkeze geri
park ediyor — yolun ortasında değiştirilen bir oda kaymış bir bakış noktası miras alamasın diye,
ki bu panoramayı fark edilir ama düzeltilemez biçimde gerdirirdi.

Hareketi istemediğini söylemiş olana (`prefers-reduced-motion`) yolculuk değil, varış veriliyor.

### Çekilmemiş odaya giden halka: gizlenmiyor, karanlığa çıkıyor

Halkaların var olma sebebi olan iş akışı şu: emlakçı daireyi dolaşır, her kapıya bir halka
bırakır, sonra dönüp odaları çeker. Henüz çekilmemişleri gizlemek bu akışı görünmez kılardı —
halkayı koyan kişi hiçbir şey görmez ve işe yaramadığı sonucuna varırdı. Bu yüzden çizilir, ama
kehribar rengiyle ve daha sönük.

Tıklandığında ne olacağı bu özelliğin en çok tartışmaya değer kararı. **Yolculuk yine yapılıyor**
ve karanlıkta bitiyor. Çünkü o kapının ardında bugün gerçekten hiçbir şey yok; karanlık, kılık
değiştirmiş bir hata durumu değil, **o kapının ardındaki şeyin kendisi.** Tur yarı çekilmişken
bile halka halka gibi davranmaya devam ediyor.

Bunun ilk hali farklıydı ve daha kötüydü: hareket yoktu, ekranın üstünde uçan bir mesaj vardı.
O hâliyle halka, tıklanınca *reddetmiş* bir düğme gibi görünüyordu.

Karanlıkta odanın adı, tek satırlık açıklama ve bir **Geri dön** düğmesi duruyor; alt araç
çubuğu da erişilebilir kalıyor, yani buradan çıkış tek bir düğmeye bağlı değil. Mesaj zamanla
sönmüyor: sönseydi kullanıcı, açıklaması olmayan siyah bir dikdörtgende ve çıkışsız kalırdı.

Hedef odanın 360'ı geldiğinde halka kendiliğinden turkuaza dönüyor ve aynı yolculuk bu kez
odanın kendisiyle bitiyor; ayrıca bir işlem gerekmiyor. Siyah karenin ardındaki her şey iki yol
için de aynı olduğundan tek bir `swapTo` paylaşılıyor — ikinci bir kopya, ikisinin zamanla
birbirinden ayrı düşmesinin yolu olurdu.

### Editörde halka tıklanmıyordu

Oynatıcıda çalışan şey editörde çalışmıyordu, ve sebebi iki ayrı hataydı.

**Birincisi, kodun kendi yorumunu yalanlaması.** Dokunma işleyicisinde "editörde de portal portal
olarak tıklanabilir kalmalı, yoksa yaptığınız turu denemenin tek yolu yayınlamak olur" diye yazan
yorumun hemen altında `editable ? null : hitTest(...)` duruyordu — yani editörde halka testi hiç
çalışmıyor, her tıklama "buraya etiket ekle" akışına düşüyordu. Halkayı yeni koymuş kişiye halka
bozuk görünüyordu.

**İkincisi, portalin editörde iki kez çizilmesi.** Ziyaretçinin göreceği zemin halkası, bir de
her hotspot'un aldığı küçük işaretçi. İkisi de portalin kendi yönüne konduğu için işaretçi tam
halkanın ortasına düşüyor, ve tuvalin üstünde bir DOM düğümü olduğu için halkaya nişan alan her
tıklamayı yutuyordu.

İkisinden birini silmek yerine işaretçi halkanın üstüne kaldırıldı. Kaldırma miktarı bir zevk
meselesi değil, halkanın kendi boyu: halkanın **ekrandaki en yüksek noktası uzak kenarıdır**,
çünkü o kenar daha uzakta ve dolayısıyla ufka daha yakın. Onu geçen bir kaldırma bütün halkayı
açığa çıkarıyor — her mesafede, ki asıl mesele bu: ayağınızın dibindeki halka geniş bir elips,
odanın karşısındaki ince bir çizgi; tek bir sabit sayı ikisini birden temizleyemez.

**Kaldırmanın geri alınması ölçülmesi gereken kısımdı.** Sürüklenip bırakılan noktadan kaldırmayı
düpedüz çıkarmak tersi işlem değil, çünkü kaldırma çözülmek istenen açıya bağlı. Tek yönlü hata:
her sürüklemede halka biraz daha ufka kayıyordu — sürükleme başına 0.66 derece, kulağa zararsız
gelen ama on dokunuşta kapı halkasını zeminde tam bir metre yürüten bir sapma; kimsenin araca
yormayacağı türden. Yerine sabit nokta yinelemesiyle çözülüyor: yirmi ardışık gidiş-dönüşte
kalan sapma en fazla 0.007 derece.

**Ve bunun ortaya çıkardığı daha eski bir hata.** Karanlık oda durumunda etiket katmanı artık
render edilmiyor (karanlıkta duracak oda yok), ve geri dönünce etiket bir daha görünmedi.
`HotspotLayer` düğümü yeniden kaydederken nesneyi koruyup DOM elemanını değiştiriyor, ama "en son
şunu yazmıştım" önbelleğini (`lastVisible`, `lastOpacity`, `lastZ`, kart tarafları) koruyordu.
Yansıtıcı da o eski değerlere bakıp kısa devre yapıyor, ekrandaki yeni elemana hiçbir şey
yazmıyordu — eleman da doğduğu gibi, gizli kalıyordu. Yalnızca benim koşullu render'ımın açığa
çıkardığı, ama **her yeniden bağlanmada** olacak bir hata; kaynağında düzeltildi.

### Ölçülmüş hata: hedefin adı görünmüyordu

Halkanın üstündeki isim etiketi işaretçiyi izliyor. İlk hali dinleyiciyi yalnızca bir halka
hover'dayken bağlıyordu. Aynı şeyin ucuz hali gibi görünüyor, değil: hover yeniden render olana
kadar işaretçi çoktan varmış ve durmuş oluyor, sonraki `pointermove` çok geç gelebiliyor — ya da
kullanıcı kıpırdamazsa hiç gelmiyor. O sürede etiket konteynerin sol üst köşesinde, kendi
`-100%` ötelemesiyle ekranın üstünde kalıyor. Tarayıcıda ölçüldü: etiket DOM'da, metni doğru,
`transform: none`. Düzeltme, konumu her hâlükârda takip etmek ve etiketi doğduğu anda
yerleştirmek.

### Halka yoksa: üstteki oda çubuğu

Halkalar emlakçının koymasına bağlı, ve her odaya bir tane koyup koridorları atlayan bir tur
gezilemez hale gelir. Bu yüzden geri düşüş kaldırılmadı: 360'ı çekilmiş oda sayısı ikiden
fazlaysa oynatıcının üstünde odaların **adlarının** yazdığı bir çubuk çıkıyor, tıklayan o odaya
geçiyor. Küçük resim değil ad, çünkü aynı dairenin iki odası küçük resimde birbirinden
ayırt edilmiyor. Tek odalı turda çubuk yok — seçecek bir şey olmayan bir seçici gürültüdür.

## 14. Kuş bakışı: kat planı

Turun içinde odadan odaya geçmenin yolu vardı (§13) ama **evin şeklini** gösteren bir şey
yoktu. Matterport bunu bir kat planı ve bir "dollhouse" ile yapıyor; ikisi de o kameranın
derinlik sensörüyle çıkardığı bir **mesh**'ten geliyor. Bizde mesh yok — telefon videosundan
saf rotasyonla dikilmiş bir küre var. O yüzden soru "aynısını nasıl yaparız" değil, şuydu:

> Bir panorama, içinde çekildiği odanın şekli hakkında ne söyleyebilir?

Cevap tek satır geometri, ve motor ona zaten iki yerde güveniyor (`ai/measure.py` tek görüntülü
ölçümde, oynatıcının `portalGeometry.ts`'i zemin halkalarında):

    ufkun altındaki bir yön zemini  h / tan(alçalma)  mesafesinde keser

Yani her azimut için **duvarın zeminle birleştiği yükseklik** bulunabilirse, her yöndeki duvar
mesafesi — odanın planı — çıkar. Bedeli iki şey: varsayılan bir kamera yüksekliği, ve o
birleşim hattını bulmak.

### Hattı bulmak: iki sınır, ve neden yüksek olan kazanıyor

Sınır, `ai/furniture_seg.segment_equirect`'in dekorasyon ve boşaltma araçları için zaten
ürettiği anlamsal yüzey haritasından okunuyor — **bilerek aynısı**, çünkü "zemin nerede"
sorusuna iki farklı cevap, planın duvarlarını o araçların koruduğu duvarlardan başka bir yere
koyardı.

Zemin sınırı tek başına yetmiyor, ve sebebi bir yatak odasının sıradan içeriği: uzak duvarın
önündeki gardırop, önündeki iki metre zemini gizler; o sütunlarda en üstteki zemin pikseli
gardırobun ayağıdır ve duvar **olduğundan yakın** çıkar. Mobilya bunu ancak bu yönde
yapabilir — zeminde durur ve kameradan uzağı örter — yani hatanın **işareti bilinir**, ve
ondan bir kural çıkar:

> Tavan hattı aynı duvarın üstündedir ve nadiren bir şey tarafından örtülür; örtülme yalnızca
> yarıçapı kısaltabildiğine göre, **iki cevaptan uzun olanı** doğru olanıdır.

Tavan hattının mesafeye dönüşmesi için tavan yüksekliği gerekir, ve o varsayılmıyor:
**ölçülüyor.** İki sınırın da göründüğü her sütunda ikisi aynı duvarı görür, dolayısıyla
`tan(tavan açısı) / tan(zemin alçalması)` kamera yüksekliği biriminde tavanın gözden
yüksekliğidir, ve böyle her sütunda **aynı sayıdır.** Kestirim bu oranın ortalamasını değil
**modunu** alıyor, çünkü anlaşmayan sütunlar gürültüden değil bir sebepten (altta gardırop,
üstte yüksek dolap) anlaşmıyor.

Boyutları tam bilinen üç prosedürel odada ölçüldü (`scripts/bench_floorplan.py`), kusursuz bir
yüzey haritası verilerek:

| kaynak | ortalama yarıçap hatası | zemin IoU |
|---|---|---|
| yalnız zemin sınırı | **88 cm** | 0.49 |
| zemin + tavan kuralı | **1.4 cm** | 0.99 |
| aynı kural, gerçek segmenter'ın maskesiyle | 3.5–14 cm | 0.88–0.95 |

Tavan kuralı bir iyileştirme değil, plan ile tahmin arasındaki farktır. Tavan yüksekliği
kestirimi aynı testte santimetre isabetli (2.89 / 2.88 / 2.80 m). Gerçek yatak odasında
yönlerin **%55'i** cevabını tavandan alıyor — ve alan yalnız zeminle 13.3 m², tavan kuralıyla
16.3 m².

### Tavan, zenit'i içeren bölgedir

İlk sürümde tavan sınırı "ufuktan yukarı çıkarken karşılaşılan ilk tavan koşusu" idi ve gerçek
bir yatak odası bunu hemen cezalandırdı: geceleyin karanlık bir pencere tavan olarak
etiketlendi, adacığı ufkun beş derece üstünde kaldı, ve dört metrelik odada **on iki metre
uzakta** bir duvar istendi. İlk çözüm bir orandı (tavan, zeminin en çok 2.5 katını söyleyebilir)
ve pahalıydı: sentetik odalarda ortalama hatayı 1.4 cm'den 15 cm'e çıkarıyordu, çünkü döşeli bir
odada gerçek örtülme rutin olarak üç-dört kat.

Doğru koruma yapısal: **bir tavan tek bir bölgedir ve kürenin tepesini içerir.** O yüzden tarama
zenit'ten aşağı iniyor ve ilk koşunun *alt* kenarı alınıyor; ufkun üstünde yüzen bir adacık
artık tavan değil. Oran kaldırıldı. İki tarama ters yönlere gidiyor ve bu asimetri kasıtlı:
zemin için istenen **en uzak** görünen zemin şeridi (ufuktan aşağı ilk koşu), tavan için istenen
gerçekten tavan olan bölge.

### Çizilen şey ölçüm değil, ölçüme oturtulan oda

İlk sürüm ölçümü çiziyordu: radyal profil, sadeleştirilmiş. Sadıktı ve kullanılamazdı. Profil
"her yöndeki en yakın şeyin mesafesi"dir; gerçek bir yatak odası bu haliyle, içinden bir düzine
küçük ısırık alınmış (tavan hattının da göremediği bir gardırop, bir perde, üst üste kutular) ve
dışına iki uzun kama çıkmış (açık kapıdan görünen öbür odanın zemini) bir amip olarak geliyor.
Kimse ona bakıp oda görmez, çünkü oda o şekilde değildir: **daireler dikdörtgenlerden kurulur.**

O yüzden çizim, ölçümden **başka bir soruyu** cevaplıyor. Ölçüm: "bu yönde en yakın yüzey ne
kadar uzakta" — `RadialProfile` içinde duruyor ve hiç atılmıyor. Çizim: "bu mesafeleri en iyi
hangi dik açılı oda açıklıyor". Ayrım önemli, çünkü ikisi farklı bozuluyor: bir örtülme ölçümün
tek bir yönünü içeri çeker ve **oturtulmuş odayı yerinde bırakır** — bir planın ihtiyacı olan
davranış tam olarak budur.

**Model bilerek sığ: bir dikdörtgen, olsa olsa bir köşesi kesilmiş.** Bu tavan aritmetiğin
sınırı değil, odalar hakkında bir iddia. Her şekli çizebilecek kadar serbest bir model, amibi
yeniden çizer.

Uyum ölçütü **kırpılmış**: hata `min(|fark|, 40 cm)`. 40 cm sapan bir yön ile dört metre sapan
bir yön aynı bedeli öder. Bütün dayanıklılık burada, ve açık kapıyı zararsız kılan da bu: kapının
ardındaki kama, metrelerce yanlış olan bir avuç yöndür, ve bedelleri kırpılınca duvarı kendilerine
doğru çekemezler. 40 cm, segmenter'ın duvar boyunca kendi saçılımının (3.5–14 cm) üstünde ve bir
duvarla karıştırılabilecek her şeyin altında.

Dört duvar, koordinat inişiyle 2 cm'lik bir ızgarada aranıyor; her duvar, ona ulaşan her yön için
diğer üçünden bağımsız olduğundan üç tur yetiyor. Odanın **yönü** önce bulunuyor: her sınır
parçası, uzunluğu ağırlığıyla, **açısının dört katında** oy veriyor — böylece çeyrek tur arayla
duran duvarlar birbirini götürmek yerine güçlendiriyor.

Köşe kesiği iki kez hak etmek zorunda: en az 1.5 m² zemin çıkarmalı **ve** uyumu en az %12
iyileştirmeli. L şeklinde bir salon gerçek bir şeydir ve böyle çiziliyor; iki testten biri
olmasa kesik, köşedeki gardırobu çizmenin bir yolu olurdu — yani uydurmanın.

Ölçüldü (gerçek segmenter maskesiyle, boyutları bilinen üç oda):

| çizim | zemin IoU | köşe sayısı |
|---|---|---|
| ölçümü olduğu gibi çizmek | 0.88–0.95 | 6–11 |
| ölçüme oda oturtmak | **0.984–0.991** | 4 |

Oturtulan odanın kenar uzunlukları gerçeğinden en fazla 9 cm sapıyor. Gerçek yatak odasında
sonuç 4.1 × 3.2 m, tek bir dikdörtgen; kapıdan giren kama artık duvarı dışarı çekmiyor, kapı
zaten kendi işaretiyle çiziliyor.

**Planın yönü de çizimin bir kararı.** Çıpa odası çerçeveyi tanımlıyor, o yüzden çizmeye değer
bir çerçeve tanımlaması sağlanıyor: kendi duvarları sayfaya dik gelene kadar döndürülüyor.
Bırakılsaydı bütün daire, bir kişinin çekime başlarken baktığı yöne göre eğik çizilirdi —
okuyucunun asla öğrenemeyeceği bir sebeple yirmi derece yatmış bir plan. Kuzey diye bir
referans olmadığından (aşağıya bakın), ilk odanın duvarları eldeki en iyi referans.

### Odaları tek bir çerçeveye koymak

Bir panorama odanın şeklini sabitler, **yerini** sabitlemez: her çekimin yaw başlangıcı
fotoğrafçının dönmeye başladığı yerdir, ve yatay düzlemde hiçbir başlangıcı yoktur. Plan için
**iki odadan da görünen aynı şey** lazım, ve turda zaten bir tane var: bir kapı.

Bir halka zeminde bir noktadır (yaw, pitch ve varsayılan kamera yüksekliği). İki oda da aynı
kapıya halka koymuşsa, o **tek nokta iki koordinatla** verilmiş demektir, ve iki koordinat bir
bağıl pozu tamamen sabitler. Çözüm bir açı çıkarması: kapı iki odada da aynı nokta olduğuna ve
odalar kapının iki yanında durduğuna göre, ikinci odanın kapıya kerterizi birincinin tersidir.
Optimizasyon yok, tek bir çıkarma var.

Tek halka iki denklem verir; bağıl pozun üç bilinmeyeni vardır. Eksik olan **karşı odanın nasıl
döndüğü**dür ve bu bilgi bu taraftaki hiçbir fotoğrafta yoktur. Emlakçının iş akışı da iki
halkayı zorunlu kılamaz — daireyi dolaşıp her kapıya bir halka bırakır, dönüş halkası ancak o
oda çekilince konur. O yüzden tek yönlü kapı destekleniyor, `estimated` işaretleniyor ve **kesik
çizgiyle** çiziliyor: oda gerçekten o kapının ardında ve o kerterizde; dönüşü modülün açıkça
yaptığı bir tahmin, mirasla gelmiş bir olgu değil. Dört aday dönüş (odanın kendi duvar ızgarası,
çeyrek tur arayla) içinden çizilmiş odaların üstüne en az binen seçiliyor, ve mesafe odanın
kendi profilinden okunuyor — yani kapı, içinde olacağı duvara yaslanıyor.

**Birbirine bağlı olmayan gruplar yan yana dizilmiyor.** İki oda birbirine bağlıysa ama ikisi de
dairenin geri kalanına bağlı değilse, geri kalana göre yerleri bilinmiyordur; onları yine de bir
köşeye koymak, kimsenin ölçmediği bir şeyi söyleyen bir çizim olurdu. Listeleniyorlar. Plan tek
parça bir şeydir, ya da daha küçük bir plan ve bir listedir.

### Son söz emlakçının: planı elle düzenlemek

Yukarıdaki üç cevap da doğru, ve üçü de daireye bakan emlakçıya yetmiyor:

- iki taraftan işaretlenmiş kapı **çözüyor**;
- tek taraftan işaretlenmiş kapı odayı doğru kerterizde koyup **dönüşünü tahmin ediyor** —
  dört ihtimalden biri, ve üçü yanlış;
- hiçbir kapının ulaşmadığı oda **hiç konmuyor**.

İkinci ve üçüncü durumda ekranda yanlış ya da eksik bir plan var ve düzeltecek hiçbir şey yok.
Bu yüzden otomatik yerleşim artık **ilk taslak**: odalar sürükleniyor, seçili oda çeyrek tur
çevriliyor, ve hiç halka konmamış bir tur bile baştan sona elle kurulabiliyor.

**Yerleştirme bir eğilim değil, o oda için cevabın kendisi.** `plan/placements.py`'deki bir
kayıt çözücünün bulduğunu ezer. Üç sonucu var ve her biri bir tur sınıfını planlanabilir
yapıyor:

*Hiçbir kapının ulaşmadığı oda konabiliyor.* Halkası olmayan turun çözülecek bir grafiği yok;
artık biri odaları sürüklediği an planı var.

*Konan oda kendi başına bir çıpa.* Yürüyüş otomatik çıpadan olduğu kadar **her konmuş odadan
da** başlıyor, yani koridoru düzeltmek koridorun açıldığı her şeyi beraberinde götürüyor —
komşularını tahmin edilmiş yerlerinde bırakmıyor.

*Düzeltme tahminden uzun yaşıyor.* Kayıtlar sahneye göre tutuluyor, çözücünün ürettiği hiçbir
şeye göre değil; odayı yeniden ölçmek ya da ötesindekini çekmek emlakçının emeğini silmiyor.

**Çeyrek tur, çünkü düzeltilecek tek şey o.** Her oda zaten kendi duvar ızgarasına oturtuluyor
ve çıpa sayfaya dik çevriliyor (§ yukarısı), dolayısıyla bütün odalar zaten eksenlere paralel.
Belirsiz olan yalnızca **hangi çeyrek** olduğu — tek yönlü kapının dört ihtimali. Serbest
açılı bir döndürme kulbu, çözülmüş olanı bozma imkânı vermekten başka bir şey yapmazdı.

**Duvarlar yapışıyor.** Sürüklenen odanın kenarı bir başkasınınkine 25 cm'den fazla
yaklaşmıyorsa aradaki fark kapatılıyor. Daire duvarlarını paylaşan odalardan oluşur; elle
dizilip birkaç santim açık bırakılmış bir plan, gözün hemen yakaladığı ince beyaz şeritlerle
bozuk görünür. Yapışma, elle yerleştirmenin kolaj değil çizim üretmesini sağlayan şey.

**Her düzeltme sunucuya gidip bütün planı geri getiriyor**, bir onay değil. Çünkü yerleştirme
yerel değil: yürüyüş her konmuş odadan yeniden başladığına göre bir odayı taşımak ötesindekileri
de taşıyabilir — taşımalı. Sürüklenen odanın yerel kopyasını yamalayan bir istemci, sunucuda
olmayan bir daire çizerdi.

**Ölçülmüş bir yarış vardı.** Her sürükleme ve her çevirme kendi isteği; FastAPI bunları bir
iş parçacığı havuzunda koşturuyor, ve birlikte gelen ikisi dosyayı biri yazmadan önce ikisi
birden okuyor — ikinci yazma, birincinin eklediğini hiç görmemiş oluyor. Tarayıcıda bu, odanın
eski yerine geri zıplaması olarak görünüyor, ki doğrudan manipülasyonla çalışan bir editörün
asla yapmaması gereken tek şey budur. On altı eşzamanlı istekle yeniden üretildi; oku-değiştir-
yaz artık tur başına tek bir kilidin altında, ve dosya yanına yazılıp `os.replace` ile yerine
taşınıyor, böylece okuyan biri onu yarılanmış halde yakalayamıyor.

Çizimde konmuş oda **düz çizgiyle** görünüyor, şekli ölçülmemiş olsa bile: bir insanın odanın
nerede olduğunu söylemesi, yazılımın tahmin etmesi değildir.

### Çizimin söylemesine izin verilenler

Üç çizgi türü, ve aralarındaki fark özelliğin dürüstlüğü:

*düz turkuaz* — odanın kendi panoramasının ölçtüğü bir duvar.
*kesik, sönük* — planda yeri olan ama ölçülmüş şekli olmayan oda, ya da dönüşü tahmin edilmiş
oda; disk "bir oda, aşağı yukarı bu büyüklükte" demektir, duvarları hakkında bir iddia değil.
Emlakçının elle koyduğu oda ise **düz** çizilir — bir insanın nerede olduğunu söylemesi,
yazılımın tahmin etmesi değildir.
*kehribar* — henüz çekilmemiş oda. Zemin halkasının aynı olgu için kullandığı **aynı sabitten**
gelen aynı kehribar (`PortalRings.PENDING_COLOUR`), ki ikisi zamanla birbirinden ayrı düşmesin.

**Pusula gülü yok, ve bu kasıtlı.** Matterport'un planında kuzey oku var, çünkü o kamera hangi
yöne baktığını biliyor. Telefon çekimi bilmiyor: motor ufku yerçekimine göre düzler, bu *yukarı*yı
sabitler ve kuzey hakkında hiçbir şey söylemez.

Telefonda bir manyetometre var ama ona güvenilmiyor, ve bu karar zaten ölçülmüş halde
`web/src/lib/capture/useOrientation.ts`'te duruyor: buzdolabının, çelik dikmeli duvarın veya
yerden ısıtmanın yanında `webkitCompassAccuracy` nominal 10–15 dereceyi bildirmeye devam ederken
gerçek sapma 30–40 derece oluyor, ve anomaliyi geçerken **sıçrıyor**. Çekim bu yüzden yaw'ı her
zaman göreli izden kuruyor. Plana kuzey oku koymak, o hattın bilerek reddettiği sayıyı bu kez
alıcının önüne koymak olurdu. Referans, planın kendi yönü — zaten izleyicinin ihtiyacı da o:
"mutfak arkamda", "mutfak kuzeyde" değil.

Bakış konisi her karede yazılıyor, pusula kadranıyla aynı döngüde ve aynı sebeple (kare hızında
bir `setState` bütün oynatıcıyı, tuvali de dahil, yeniden çizerdi). Koninin açısı kameranın
yaw'ına odanın kendi dönüşünün eklenmesidir; toplamayı unutan bir uygulama **tam olarak
geliştiricinin ilk açtığı odada** doğru çalışır, çünkü çıpa odasının dönüşü sıfırdır. Testte
yazılı olmasının sebebi bu.

**Ölçek bir varsayım** ve plan bunu yazıyor. `plan/types.CAMERA_HEIGHT_M` bilerek oynatıcının
`EYE_HEIGHT`'i (1.60 m), `ai/measure`'ın 1.55'i değil: aynı fiziksel tahminin iki yuvarlaması,
ve bu taraf oynatıcıyla aynı olmak zorunda, çünkü plan bir kapıyı zemin halkasının durduğu yere
çiziyor, ve %3'lük bir uyuşmazlık tek iddiası "bu ikisi aynı kapı" olan bir çizimde ikisini bir
el genişliği ayırırdı.

### Oynatıcıda plan nerede duruyor: sol altta, resmin üstünde değil yanında

İlk sürümde plan düğmesi alttaki araç çubuğunun içindeydi ve tıklayınca **bütün kareyi
kaplıyordu**. İkisi de yanlıştı, ve ikinci olan pahalıydı.

Araç çubuğu tek atımlık kontrollerden oluşuyor: hangi oda, ne yön, hangi çözünürlük, tam
ekrana geç. Plan onlardan biri değil — **açılan**, açık kalan, ve bir yere yaslanması
gereken bir şey. Ekranın ortasındaki bir hapın yaslanacağı bir kenarı yok, o yüzden paneli
ortadan açmak ve odayı kapatmak zorunda kalırdı. Köşeye çakılınca panelin bariz bir çıkış
noktası ve bariz bir geri dönüşü oluyor.

Kareyi kaplamanın bedeli ise çizimin en iyi yanını sessizce çöpe atmasıydı: **bakış konisi
kamerayı canlı izliyor.** Resim kapalıyken döndürecek bir şey kalmadığından canlı koni yalnızca
bir süstü. Köşeye yerleşince olması gereken şeye dönüşüyor — panorama yanında sürüklenmeye
devam ederken bakılan **harita**, ve siz döndükçe süpüren ok. Ölçüldü: panel açıkken küreyi
çevirmek koniyi 21.5°'den 54.5°'ye taşıyor ve pusula kadranı aynı açıyı ters işaretle
gösteriyor — ikisi de aynı kare döngüsünden yazılıyor.

Açmanın maliyeti de düştü. Tam ekran plan bir **mod**dur: odadan çıkarsın, bakarsın, geri
dönersin. Panel bir **bakış**tır.

Kapatmanın üç yolu var ve bu kasıtlı: düğme aç/kapa çalışıyor, panelin kendi kapatma düğmesi
var, ve `Esc` iş görüyor. Birinin fotoğrafının bir kısmını kapatan bir panel, çıkışı aranan bir
şey olmamalı.

Panel yalnızca **arkasında çizecek bir şey varken** beliriyor — ölçülmüş tek bir odası bile
olmayan turda düğme hiç çıkmıyor. VR'ın kuralı ile aynı: yapabileceği bir şey varken sunulur,
yokken hiç görünmez.

**Sahne editörü de bu düğmeyi alıyor**, ve bir süre almıyordu: koşulda `planAvailable`'ın
yanında `showChrome` de vardı, o da sessizce "editörde plan yok" demek oluyordu. Oysa yanındaki
odanın nerede olduğunu bilmenin en çok işe yaradığı yer orası — planda bir odaya tıklamak aynı
zamanda o odanın editörüne gitmenin yolu. Editör planı **yalnızca okuyor**: `plan.get`, diskten
aritmetik, milisaniye. Ölçme ve elle yerleştirme ev sayfasındaki `PlanPanel`'de kalıyor, çünkü
o oda başına bir sahne ayrıştırma geçişi ve saniyeler. Planı olmayan evde istek boşa düşüyor,
oynatıcı çizim bulamıyor, düğme çıkmıyor — ve emlakçıya her oda açtığında bir hata balonu
gösterilmiyor.

### Sol alt köşenin tek sahibi olmak zorunda: `PlayerDock`

Köşeyi iki ayrı bileşen sahipleniyordu — plan yuvası ve araç rayı — her biri kendi
`absolute bottom-0 left-0`'ı ile. Bu, ikisi asla aynı anda çıkamadığı sürece doğruydu; gömülü
oynatıcıya **ölçme** izni verildiği gün öyle olmaktan çıktı. Kat planı olan bir ilan sayfasında
"Kuş bakışı" hapı ile "Ölç" hapı **birebir aynı koordinatta** çiziliyordu (ikisi de x=12,
y=852). Böyle bir hata kendini haber vermiyor: iki eleman da tek başına doğru, sadece köşe
yanlış.

Artık köşenin tek sahibi var, ve köşedeki **her düğme tek bir satırda**:

```
[beta] [Boşalt] [Dekor]  │  [Kuş bakışı] [Ölç] [Yatırım]  │  [⟳]
 resmi yeniden yazan        resmin üstünde bir şey açan       davranışı değiştiren
```

Bu, ilk denemenin düzeltmesi. İlk denemede üst üste binme kalktı ama köşede **iki cam levha
üst üste** kaldı: üstte yapay zekâ araçları, altta görüntü kontrolleri, ve etiketleri farklı
uzunlukta olduğu için genişlikleri de farklı. Kulağa geldiğinden kötü. Köşedeki bir istifin
yaslanacağı düz bir kenarı yok — her satır aynı sol marjdan başlıyor ve yazısı nerede bittiyse
orada bitiyor, göz de bir kontrol değil tırtıklı bir merdiven okuyor. Üstelik istiflemenin
kendisi bir hiyerarşidir: kimsenin duyurulmasına ihtiyaç duymadığı bir ayrımı duyururken
düzenin tek düz çizgisini harcıyordu. Ayrım kıl payı ayraçlarda yaşamaya devam ediyor —
gerçekten değdiği fiyata.

Sığmadığında **kaydırıyor, sarmıyor**: sarmak köşeyi tam da olmaması gereken şeye, farklı
genişlikte iki satıra geri döndürürdü. 320 px'lik bir ekranda çubuk 296 px'e sabitlenip
kayıyor, tek satır kalıyor; 375 px'te zaten tamamı sığıyor.

**Otomatik dönüş de araç çubuğundan buraya taşındı**, ve iki şey birden değişti. Yeri: bir
davranış anahtarıydı, anlık ve geri alınabilir görüntü kontrollerinin arasında etiketsiz bir
ikon olarak ekranın en zor bulunan yerinde duruyordu. Zamanlaması: sürüklenme "dört saniye
kimse dokunmazsa" kapısına bağlı — kendi kendine başladığında doğru olan bu kural, **düğmeye
basıldığında yanlış**. Basış girdi sayılmıyor (kontrol `data-player-ui` taşıyor, işaretçi
işleyicileri onu atlıyor), ondan önceki sürükleme ise sayılıyor: bas, ve resim dört saniyenin
kalanı boyunca ölü kalsın, sonra bir buçuk saniye daha rampada. Ekranda bunu bozuk bir
anahtardan ayıran hiçbir şey yok. Artık basış boşta-kalma saatini geriye alıyor ve rampa o an
başlıyor.

Hız da bunun için yeniden ayarlandı. 1.6°/s ve 1400 ms'lik rampa **kimsenin istemediği** bir
sürüklenme için seçilmişti — göz ucuyla fark edilsin diye. İntegrali alınınca ilk saniyede
0.51°, ilk üç saniyede 3.67°: bir oda fotoğrafında hiç.

| | ilk 1 sn | ilk 3 sn | tam tur |
|---|---|---|---|
| 1.6°/s, 1400 ms | 0.51° | 3.67° | 225 sn |
| 4°/s, 600 ms | 2.77° | 10.77° | 90 sn |
| **6°/s, 600 ms** (sevk edilen) | **4.15°** | **16.15°** | **60 sn** |

Ölçek dakika: 6°/s'de çalışır bırakan biri odanın tamamını dakikada bir görüyor, ki bir
panoramada geçirilen süre aşağı yukarı budur. Rampa duruyor — başlangıç yine de yumuşak
olmalı, sadece artık bakmayan birine göre ölçülmüyor. Aritmetik `autoRotateDelta`'da, testler
onu entegre edip dereceyi okuyor.

### Okumak bedava, ölçmek değil

`GET /properties/{id}/plan` diskteki profillerden planı kuruyor: saf aritmetik, milisaniye, hiçbir
model yüklemiyor. `POST /properties/{id}/plan/build` ölçülmemiş odaları ölçüyor: oda başına on altı
pencerelik bir sahne ayrıştırma geçişi, bu kartta ~12 saniye.

Bu ayrım, üçüncü taraf bir emlak portalına gömülü oynatıcının planı gösterebilmesinin sebebi:
istek üzerine hesaplanan bir plan, yabancının sayfasını başkasının GPU'sunda bekletirdi.
Yayımlanmış bir plan ise bir çizimdir, ve çizim ucuzdur. Maliyet de kararın olduğu yere düşüyor:
daireyi çekmeyi bitiren emlakçı bir kez "ölç" der, sonucu görür, yeniden işlemeden sonra tekrar
basar.

Profil **kamera yüksekliği biriminde** saklanıyor, metre değil, ve sahnenin kendi dizininde
`layout.json` olarak duruyor (kimliğine değil, panoramanın `mtime`'ına göre bayatlıyor).
Boyutsuz tutmak düzenlilik değil: yükseklik bu özelliğin sonradan düzeltilmesi en muhtemel
sayısı, ve metre cinsinden bir önbellek o düzeltmeyi uygulamak için oda başına bir segmentasyon
daha isterdi.

`Scene`'e bir kolon değil bir dosya, çünkü diğer zengin alt nesnelerin (§12) aksine bunu yazan
geçiş **hiç koşmayabilir** — segmenter'ı olmayan bir kurulumda hiç üretilmez — ve odanın gayet
iyi yaşayabildiği bir alan için şema değişikliği ödemeye değmez.

**Bu da bir iş (job) olmalı, henüz değil** — `/ai/empty-room` ve dekorasyon rotalarıyla aynı
borç, aynı sebeple (§11).

### Bugün yapamadıkları

- **Ayna bir odadır.** Buradaki hiçbir şey ikisini ayırt edemez, ve kapıdan farklı olarak bir
  ayna, uyumun inanacağı kadar büyük olabilir.
- **Dikdörtgen de L de olmayan bir oda, ikisinden yakın olanı olarak çiziliyor.** Bu tavan
  kasıtlı; her şekli çizebilen bir model ölçümün gürültüsünü çizer.
- **Kapılar tespit edilmiyor.** Tek yönlü bir halkanın karşı odadaki karşılığı, o odanın kendi
  panoramasındaki kapıyı bulabilseydi (`ADE20K`'da kapı bir sınıf) tahmin değil çözüm olurdu.
  Tek yönlü bağların doğru **otomatik** çözümü orası; bugünkü cevap, emlakçının çeyrek turu
  kendisinin düzeltmesi.
- **Tek kat.** Bir merdiven yalnızca bir kapıdır; kat kavramı yok.

## 15. 3B gezinti: videodan Gaussian splat tura

Buraya kadar her şey "odanın **ortasından** nasıl görünüyor" sorusuna cevaptı: bir küre, bir
nokta. Bu bölüm başka bir soruya cevap: "odanın **herhangi bir yerinden** nasıl görünüyor".
Ziyaretçi yürüyor, eğiliyor, masaya yaklaşıyor, kapıdan öteki odaya geçiyor; ve bunun için
telefon dışında hiçbir donanım yok. Hedef, splattour.com'un yaptığı şeyin aynısı; bileşenler
ve kararlar onların ürününün dışarıdan ölçülebilen şekline göre seçildi (CDN'deki tur
yapılandırması, `.sog` dosyalarının içi, oynatıcının konsolu — bkz. `engine/biosphere/splat/`
modül başlıkları).

```
genel gezinti videosu ─► kareler ─► öznitelikler ─► SfM ─► düzleme + ölçek ══ TUR ÇERÇEVESİ
                                                                                   │
oda videosu ─► kareler ─► öznitelikler ─► SfM (+ ödünç genel kareler) ─► Sim3 ──────┤
                                                                                   │
              her alan: undistort ─► Brush (3DGS) ─► temizlik ─► SOG + küçük resim + zarf
                                                                                   │
                                            hepsinin kat planı ─► tour.json (manifest)
```

### Çekim protokolü, ve neden bir "genel gezinti" var

SplatTour'un kayıt rehberi birebir alındı çünkü her maddesi bir hatanın cevabı: dikey telefon,
ultra-geniş lens (0.5×/0.6×), AE/AF kilidi, saniyede bir adım, dönüşlerde dur-dön-devam, oda
başına 2–4 dakika ve **dört geçiş** (çevre bir yön, çevre öbür yön, çapraz, detaylar). Bunun
üstüne bir de **genel gezinti**: kapıdan girip bütün daireyi dolaşan tek bir 1–3 dakikalık
video. Odalar ayrı ayrı çözülüyor (yüzlerce kare, dakikalar); ama ayrı çözülen iki oda
birbirinin nerede olduğunu bilmez. Genel gezinti, her odanın **kapı eşiğini** görmüş olan tek
kayıttır, ve odalar ona **kayıt** edilerek tek bir çerçeveye oturur (§ "kayıt" aşağıda). Tek
odalı bir tur için genel gezinti gerekmez: oda kendi çerçevesidir.

### Kareler (`frames.py`)

Üç dakikalık bir oda videosu 5.400 karedir; bir splat için birkaç yüz **keskin ve
birbirinden farklı** kare yeter. ffmpeg 6 fps'e indirir (ekran döndürme matrisi uygulanarak,
uzun kenar 1920'ye kırpılarak), her zaman penceresinden Laplacian varyansı en yüksek kare
seçilir (saniyede 2, en az 80 en çok 420). Ayırt edilemeyen ardışık kareler (32×24 küçük
resmi bir öncekiyle aynı: yürüyen durmuş) pencereler kesilmeden **önce** tek bir aday olarak
birleştirilir — sonra birleştirilseydi bir pencere hep bir önceki pencerenin tuttuğu karenin
kopyasını seçip atıyor, kare bütçesi kameranın durduğu yerde eriyordu. Keskinlik ölçüsü
panorama motorunun kullandığı ölçü; mutlak anlamı yok, sıralaması var.

### Öznitelikler ve eşleştirme (`features.py`)

Odalar SIFT'in en kötü durumudur: boyalı duvar, tekrarlayan fayans, cam. Bu yüzden
öznitelikler **ALIKED** (Apache-2.0 ağırlıklar), eşleştirme **LightGlue**; ikisi de torch ile
GPU'da, `lightglue` paketinden. pycolmap'in kendi ALIKED/LightGlue yolu Windows wheel'inde ONNX
olmadığı için **süreci abort eder** (istisna değil, abort) — o yüzden eşleşmeler hloc'un
yaptığı gibi COLMAP veritabanına **içe aktarılır** ve pycolmap yalnızca en iyi olduğu işi
yapar: geometrik doğrulama, haritalama, bundle adjustment.

Çiftler: her kare sonraki 12 kareyle (ve 2^k uzaklıktakilerle, COLMAP'in sıralı eşleştiricisi
gibi) eşleştirilir; **döngü kapanışı** için her karenin **DINOv2** (ViT-S/14, CLS + patch
ortalaması, torch.hub'dan bir kez iner) gömmesine en yakın 8–12 uzak kare aday olur, LightGlue
doğrular ya da reddeder. Havuzlanmış yerel tanımlayıcı + küçük resimden oluşan ucuz
tanımlayıcı yedek olarak kalır (çevrimdışı), ama bir dairenin tekrarlayan beyaz duvarlarında
DINOv2'nin yanına yaklaşamıyor. Tutan döngü çiftleri (≥ 60 iç nokta) komşularını da çeker:
`(i±2, j±2)` ikinci turda eşleştirilir. Tek bir şanslı erişim isabeti iki geçiş arasında tek
bir kenardır; global haritalayıcının bir geçişi ötekinin üstüne **katlamaması** için bir kenar
değil bir bant gerekir. Birden çok video kendi içinde sıralı, birbirleriyle yalnızca erişimle
eşleştirilir — odalar genel gezintiyle böyle buluşur.

Anahtar nokta konvansiyonu: LightGlue piksel merkezini (0,0) sayar, COLMAP (0.5,0.5); içe
aktarırken +0.5. Tanımlayıcılar kullanımlar arasında CPU'da durur; bir turun altı videosu ve
eğitim için gereken GPU aynı karttır.

### Kopuk zincir: sondalama, köprüleme, bölme (`chain.py`)

İlk gerçek telefon videosu — 82 saniyede dolaşılmış, bembeyaz duvarlı **boş** bir daire,
WhatsApp'tan geçmiş 576×1024 — hattın bench'te görünmeyen zayıf noktasını gösterdi. Kare
başına 446 öznitelik, görüntü başına 30 3B nokta ile haritalayıcı aç kaldı; eğitim
tutarsız geometriye oturdu (PSNR 17,8; splatların %44'ü saydam), görüntüleyicide ev
değil bulamaç vardı. Ölçüldü ve sırayla düzeltildi:

1. **Yoğunluk.** Saniyede 1,5 yerine 3 kare; ALIKED eşiği 0,1 → 0,02 ve 3–4K nokta. Aynı
   videoda kare başına 446 → 1.439 özellik, doğrulanmış çift 528 → 2.635, zincir kopuğu
   25 → 13; nokta 3.218 → 17.461. Dokusuz odada bu, haritalayıcının eline geçen her şey.
2. **Güvenilirlik filtresi** (`unreliable_cameras`): 60'tan az noktayla sabitlenmiş ya da
   iki komşusundan da 8 adım uzağa fırlamış kareler eğitime girmez. Tutulan karelerde
   PSNR 17,7 → 20,0.
3. **Zincir sondası.** Eşleştirmeden önce her ardışık kare çifti LightGlue + temel
   matrisle sınanır; 30 iç noktanın altındaki bağ **köprülenir**: kaynak video o boşlukta
   **videonun kendi kare hızında** (30 fps) yeniden okunur ve zincir bu karelerden geçerek
   tutuyorsa kareler kalır. Köprü içindeki eşik daha düşüktür (18): ardışık iki köprü
   karesi 33 ms arayla çekilmiştir, geometrileri neredeyse aynıdır ve zayıf bir bağ orada
   da zinciri taşır. Ölçüm (aynı video, 13 kopuk):

   | kopuk | 10 fps en zayıf bağ | 30 fps en zayıf bağ | karelerin keskinliği | sonuç |
   |---|---|---|---|---|
   | 8 tanesi | ≥ 30 | ≥ 30 | 13–340 | 10 fps'te kapandı |
   | 2, 4, 180, 181 | 11–29 | **20–31** | 1,2–13,3 | 30 fps + eşik 18 ile kapandı |
   | 182, 216, 258 | 0 | **0–7** | **0,9–1,5** | kapanmıyor |

   Son üç kopuk kamera kapıda savrulurken **bulanmış** karelerdir (Laplacian varyansı ~1;
   tutulan bir karede ~37): ne kare hızı ne de yoğun eşleştirici (LoFTR de sıfır verdi)
   orada bir şey bulabilir. Onlar kesik olarak kalır.
4. **Bölme.** Kapanmayan kopuklar alanı **segmentlere** böler; en büyüğü alanın adını
   taşır, diğerleri odalar gibi tura katılır ("Genel gezinti · 2"): çapalarla
   kaydolabiliyorsa yerleşir, kaydolamıyorsa kendi zemininde düzlenip `placed: false`
   ile ada göre açılan bir alan olur. Haritalamadan **sonra** ikinci bir kesik daha var
   (`split_at_jumps`): ardışık iki kare arasında hiçbir yürüyenin atamayacağı bir adım
   haritalayıcının iki parçayı yanlış diktiği yerdir; model orada kesilir, parçalar yine
   ayrı alan olur. Bir bulamaç asla teslim edilmez; ada göre açılan bir oda odadır.

   Bu testlerin ölçüsü **saniyedeki metredir, kare indeksindeki değil** — ve bu, köprüleme
   eklendiğinde öğrenilen bir derstir. Köprü kareleri 33 ms arayken seçilen kareler 330 ms
   aralıklıdır; indeks üzerinden alınan bir medyan yoğun bölümlerin eline geçer ve **normal
   bir yürüme adımı** on kat sıçrama gibi görünür. Ölçüldü: 180 karelik bir yürüyüş 69
   kareye indi, parçalar kendi içinde tekrar parçalandı. `outlier_cameras`,
   `unreliable_cameras` ve `split_at_jumps` artık kare adlarına karşılık gelen zaman
   damgalarını alır ve hızı (m/s) karşılaştırır; iki kare hızı da aynı ölçüyle yargılanır.

5. **Çapa havuzu.** Bir segment genel gezintinin tuttuğu parçayla değil, **daha önce
   yerleşmiş başka bir parçayla** örtüşüyor olabilir — tek geçişlik bir yürüyüşte koridoru
   ikinci kez gören parça üçüncü parçadır, birinci değil. Kayıt bu yüzden yalnızca tabana
   değil, o ana kadar tur çerçevesine yerleşmiş **bütün** alanlara karşı yapılır: çapa
   kareleri hepsinin havuzundan seçilir, hizalama hepsinin pozlarına bakar.
6. **Çekim uyarıları.** 720p altı çözünürlük, 3 Mbps altı bit hızı (mesajlaşma
   uygulaması), 1 m/s üstü yürüyüş hızı manifestte uyarı olarak taşınır ve tur
   sayfasındaki 3B kartında görünür; sonucun neden yumuşak olduğu kullanıcıya söylenir.

Bu videodan çıkan dürüst sonuç: tutarlı ve gezilebilir ama yumuşak parçalar. SplatTour
keskinliği için 1080p+ orijinal dosya, oda başına 2–4 dakika ve kapılarda yavaş dönüş
gerekir; telefon çekim modu bunları dayatır, yüklenen bir dosyada bunlar uyarı olur.

### Kamera çözümü (`sfm.py`)

pycolmap 4'ün içinde iki haritalayıcı var: artımlı (COLMAP) ve global (GLOMAP,
`global_mapping`). Global uzun dizilerde dakikalarca hızlı ve artımlı zincir gibi
sürüklenmez; ama zayıf bir görüş grafiğinde ya bütünüyle vazgeçer ya da — daha kötüsü —
**tutarlı görünen yanlış** bir model verir: yürüyüşün bir bölümünü ötekinin üstüne katlar,
yeniden yansıtma hatası yine 1 px'tir. Sıra: önce global; kabul için üç sınav: kaydettiği
kare oranı (≥ %85), yürüyüşten kopuk kamera sayısı (≤ 2, `outlier_cameras`) ve **çelişki
dedektörü** (`strong_pair_violations`): ≥ 80 doğrulanmış eşleşmesi olan iki kare aynı
yüzeye bakıyordur, dolayısıyla birbirinden 120°'den fazla dönmüş ya da yürüyüşün 25
adımından (ve kamera bulutunun yayılımından) uzak olamaz; güçlü çiftlerin %3'ünden fazlası
çelişiyorsa model reddedilir ve artımlı denenir; ikisi de çelişiyorsa daha az çelişen
kalır. Sınırlar yer gerçeğinden kalibre edildi: doğru bir modelde bu ölçüyle çiftlerin
%0–0,3'ü işaretleniyor (100°'lik lens aynı duvarı 95°'ye kadar farklı yönlerden, 4 m'den
eşliyor); ilk elden konan 75° / 12 adım sınırı doğru bir odayı %7,5 ile "katlanmış" sayıp
boşuna artımlıya gönderiyordu. Dürüst not: bu dedektör bir geçişin ötekinin üstüne
**bindiği** (yanlış ama güçlü eşleşme) durumu yakalar; bench'teki kapı geçişi katlanmaları
ise *hiç* güçlü çift olmayan yerde, zayıf döngü kapanışlarıyla oluşur ve bu ölçüde
görünmez (kalibre sınırlarla katlanmış modellerde de %0,1). Onların tek gerçek ilacı
kesintisiz video. Artımlı haritalayıcı yavaştır (300 karede ~3 dk) ama görüntüleri tek tek
PnP ile ekler; zayıf bir birleşim yerini katlamak yerine parçalar. En çok kareyi kaydeden
model kazanır, çünkü karelerin yarısından eğitilmiş splat odanın yarısı eksik splattır.

Kamera modeli **OPENCV** (tek kamera, bütün kareler). Telefonun ISP'si ultra-genişin
görüntüsünü ~75°×105°'lik (dikey) **rektilineer** bir kare olarak verir; bu modelin tam
alanıdır. Gerçek balıkgözü (≥ 150°) bu modelle **çözülmez** — polinom radyal bozulma
tanjantı takip edemez, kenar eşleşmeleri ya atılır ya da pozları 2–5° büker; ilk bench
bunu öğretti (aşağıda). Eğitim için görüntüler COLMAP'in undistorter'ıyla **pinhole**'a
yeniden örneklenir (siyah köşe bırakmadan kırparak) — Brush pinhole rasterize eder.

### Bench: gerçek bir daire, yer gerçeğiyle (`scripts/bench_walkthrough.py`)

Ölçüm veri seti ZipNeRF **berlin**: 1.525 fotoğraf, COLMAP yer gerçeği. İki tuzağı var ve
ikisi de bir gün yedi:

- Fotoğraflar **160° balıkgözü** (OPENCV_FISHEYE). Ham hâliyle pipeline'a verilince
  kamera modeli ölçülür, pipeline değil: odalar 2–3° dönme hatasıyla çözüldü, splat
  bulanık çıktı. `scripts/prepare_berlin_bench.py` her fotoğrafı yer gerçeği iç
  parametreleriyle telefonun kaydettiği kareye — dikey 1080×1920, f = 800 px (68°×100°) —
  yeniden yansıtır ve aynı pozlarla PINHOLE bir yer gerçeği yazar (`berlin_rect/`).
- Çekim bir yürüyüş değil: 0.5 m'de bir ~3 fotoğraflık **patlamalar** (10° yaw), daire
  birkaç kez dolaşılmış; hiçbir ardışık 480 fotoğraf dairenin yarısından fazlasını görmez.
  Bench videoları buna göre kesilir (`SPLITS`): genel gezinti = en çok zemin gören 480'lik
  dilim (700–1180), odalar = zeminini onun gördüğü iki ayrı dilim (540–700, 1200–1360);
  her fotoğraf 0.5 s gösterilir, seçilen karenin fotoğrafı `floor(2t)`'dir. Genel gezinti
  diliminin içinde iki **kapı geçişi** var (792 ve 1078. fotoğraflar: 1.5 m / 2.3 m sıçrama,
  ikincisinde 158° dönüş, arada tek bir eşleşme yok): bir telefon videosunda olmayan, ama
  her haritalayıcıyı zincir yerine döngü kapanışlarına mahkûm eden kopukluklar. Bench bu
  yüzden bir stres testidir; sürekli parçaların hatası ayrıca raporlanır.

Ölçüldü (rektifiye berlin, `fast` modu, RTX 5050 dizüstü; poz hatası bench'in genel
gezintiye oturttuğu tek Sim3 üzerinden, odalar için kayıt hatasını da içerir):

| alan | kare | haritalayıcı | konum hatası (dairenin uzun kenarına oran) | dönme hatası | eğitim PSNR | eğitim |
|---|---|---|---|---|---|---|
| genel gezinti (2 kapı geçişli) | 300 | artımlı | medyan %0,06; iki uç parça (0–57, 237–299) katlanmış | medyan 0,18°; katlanan parçalar 175° | 21,4 | 347 s |
| oda 1 (başında 1 geçiş) | 121 (+24 çapa) | artımlı | medyan %0,07; ilk 20 kare katlanmış | medyan 0,24° | 23,5 | 296 s |
| oda 2 (kesintisiz) | 120 (+24 çapa) | artımlı | medyan %0,04, 120/120 kare %2'nin altında | medyan 0,11°, en kötü 0,3° | 26,5 | 333 s |

Toplam 28,6 dk (kare seçimi 14 s, eşleştirme+SfM 12 dk, eğitim 16 dk, dışa aktarma 1 dk);
teslimat 1,26 M splat / 16,3 MB. Oda 2 genel gezintiye 24 çapa karesiyle oturdu (çapaların
23'ü %2 altında); oda 1'in çapaları genel gezintinin katlanmış ucuna düştüğü için yükseklik
denetimi onu reddetti ve `placed: false` ile teslim edildi — tasarlandığı gibi.

Rektifikasyondan **önce**, aynı odalar OPENCV modeliyle: konum %2–3 / dönme 2–5° (ölçek:
dairenin uzun kenarına oran); rektifikasyondan **sonra** oda 2: %0.03 / 0.08°. Kamera
modelinin doğru olması eşleştiriciden de haritalayıcıdan da daha çok şey değiştirdi.

Sentetik `synth` odası SfM'i yargılamak için **kullanılmaz**: karşılıklı duvarları özdeş
çizgili olduğu için her yöntem tutarlı ama yanlış bir çözüme oturuyor (1.3 m ortalama hata,
0.85 px yeniden yansıtma). Gerçek odalar bu kadar simetrik değildir.

### Düzleme ve ölçek (`level.py`)

SfM'in çerçevesi rastgele döndürülmüş, ilk kameranın olduğu yerde başlayan ve bilinmeyen
bir katsayıyla ölçeklenmiş bir çerçevedir. Tur için üç şey gerekir ve üçü üç ayrı ipucundan
gelir:

- **Yukarı**: önce kameraların kendi yukarı eksenlerinin ortalaması (telefon dik tutuluyor;
  yüz kare tek karenin eğimini ortalar); sonra **zemin düzlemi** ile düzeltilir: kameraların
  altındaki noktalar arasında RANSAC ile bulunan baskın düzlem, normali kamera önceline 20°
  içinde olmak şartıyla. Öncel duvarın ya da masanın zemin sanılmasını engeller, RANSAC
  yürüyenin hafif aşağı bakmasının eğimini siler. Eşik ölçeğe göre: kamera yüksekliğinin %2'si.
- **Zemin yüksekliği**: o düzlemin ofseti. Tur çerçevesinde zemin `y = 0`, yukarı `−Y`
  (§1'in konvansiyonu; oynatıcı X etrafında yarım tur döndürür).
- **Ölçek**: bilinen bir kamera yüksekliğinin (çekim talimatı "göğüs–baş hizası", 1.45 m
  varsayılır) kameraların zemine olan medyan uzaklığına oranı. ±%10 kadar hata payı var;
  ölçülü kat planı için yetmez, yürüyüş hissi için yeter. Metrik bir derinlik modeli (MoGe-2,
  MIT) aynı yere takılabilir; `estimate_leveling(metric_scale=...)` o kapıyı açık bırakır.

### Kayıt: oda genel gezintinin çerçevesine (`register.py`)

Odanın veritabanına, odanın karelerine en çok benzeyen 24 genel-gezinti karesi katılır (ilk
%35'i — kapı eşiği — daha ağırlıklı oy verir). Oda haritalayıcısı o kareleri de pozlar; genel
model onların yerini zaten bilir; iki poz kümesi Sim3'ü verir. pycolmap'in
`align_reconstructions_via_reprojections`'ı **kullanılmaz**: görüntüleri ada göre eşleyip
sonra id'lerinin de eşit olmasını CHECK'ler, id'ler hiçbir zaman eşit değildir (odanın
veritabanı birden numaralar) ve başarısız CHECK süreci öldürür. Onun yerine düz numpy: dönme
ortak karelerin **yönelimlerinden** (her çift bir R önerir; kordal ortalama), ölçek ve öteleme
merkezlerden, ikisi de RANSAC'la. Yönelimler tek başına merkezlerin çözemediği durumu çözer:
bir kapı eşiği bir doğrudur ve bir doğru yuvarlanmayı sabitlemez. Eşik yine ölçeğe göre
(hedef yayılımının %4'ü, en az 5 cm).

Kayıt tutmayan oda yine de teslim edilir: kendi başına düzlenir ve `placed: false` işaretiyle
oynatıcıda yürünerek değil **sıçranarak** açılır. Büsbütün başarısız oda raporlanıp atlanır;
tur atlanmaz.

### Eğitim (`train.py`)

Eğitici **Brush** (github.com/ArthurBrussee/brush, Apache-2.0): wgpu üstünde 3DGS, CUDA
araç seti yok, derleyici yok, tek `.exe`. Sebep düz: bu dizüstünde nvcc de MSVC de yok
(ve yönetici hakkı da yok), ve üretim makinesinde de olmayacak. Brush'ın "MCMC benzeri"
yoğunlaştırması ve sert splat tavanı (`--max-splats`) odayı SOG bütçesinin istediği ~1M
splatta tutar. Referans sonuçları (Mip-NeRF 360, 30K adım) gsplat ve INRIA'nın önünde: PSNR
29.01 / SSIM 0.875, yarı splat sayısıyla.

Bilinmesi gerekenler: v0.3.0 ikilisi **çıktısı boruya bağlıyken hiçbir şey yazmaz** (indicatif
terminal görmeyince susar). O yüzden ilerleme, `--eval-save-to-disk` ile yazdığı
`eval_<adım>/` klasörlerinden okunur ve PSNR o render'lardan **kendimiz** hesaplarız
(veri setindeki görüntülere karşı). `main` dalı bayrakları yeniden adlandırdı
(`--total-steps` → `--total-train-iters`); sarmalayıcı `--version` sorup tanımadığı sürümü
reddeder. Pozlama telafisi (bilateral grid) hiçbir sürümde yok — AE kilidinin çekim
talimatında olmasının sebebi bu.

Kalite modları (`types.SplatQuality`): hızlı 12K adım / 1280 px / 0.8M splat / SH 1;
dengeli 30K / 1600 px / 1.5M / SH 2; stüdyo 40K / 1920 px / 2M / SH 3. Teslimatta SH bantları
düşürülür (dengeli: yalnızca SH0, SplatTour'un sattığı da bu; stüdyo: 1 bant) — iç mekânda
görüşe bağlı renk azdır ve SOG boyutu iki katına çıkar.

### Temizlik ve dışa aktarma (`clean.py`, `export.py`)

Eğitilmiş bulutta üç tür çöp vardır: opaklığı 0.04'ün altındaki hayaletler, en büyük ekseni
1.2 m'yi geçen ve yanından yürüyünce odayı kaplayan devler, ve kameraların hiç yürümediği
yerde hiçbir şeyin kısıtlamadığı splatlar (kamera yolu çevresinde 6 m yatay / 3.2 m yukarı /
zeminin 35 cm altına kadar bir kutu). Sonra `@playcanvas/splat-transform` ile `.sog`:
splat başına ~10–13 bayt (PLY'nin ~250'si), oynatıcının kendi yükleyicisinin okuduğu format,
SplatTour'un dosyalarındaki üretici damgasının ta kendisi.

Alan başına üç resim daha gerekir ve splat-transform'un rasterizer'ı kamerayı bulutun **içine**
koyunca boş görüntü döndürür — odaya bakılan tek yer orası. O yüzden `raster.py`: saf torch'ta
3DGS ileri modeli (EWA projeksiyon, 0.3 px genişletme, önden arkaya
`Σ c_i α_i Π(1−α_j)`), karo yok, her (Gaussian, piksel) çifti sayılıp piksel bazında sıralanır ve
geçirgenlik çarpımı segment başına dışlayıcı kümülatif `log(1−α)` ile katlanır. Sentetik odada
Brush'ın kendi render'ına 35–40 dB, yer gerçeğine 31–34 dB (Brush'ın eval ortalamasıyla aynı);
kare başına 0.1–0.7 s. Ürettikleri: giriş pozundan **küçük resim**, oda merkezinden küçük bir
**zarf panoraması** (splatın delik bıraktığı yerde — pencere, arkanızdaki kapı — arka plan) ve
tavanı 2.05 m'de kesilmiş **kat planı** (bütün alanların yukarıdan ortografik render'ı, ev
maketi gibi; SplatTour nokta yoğunluğu çiziyor, bu daha okunaklı).

### Manifest ve oynatıcı

`tour.json` (`bundle.py`): `base` (genel gezinti; hiçbir oda yüklü değilken çizilen alan) ve
`rooms`, her biri `.sog` URL'si, bayt, splat sayısı, küçük resim, zarf panoraması, 64'e
inceltilmiş kamera yolu, ağırlık merkezi, giriş pozu (ilk kare: konum + yatay bakış), yürünebilir
kutu (kamera yolu ± 1.2 m; tavan 2.7, zeminin 30 cm altı); kat planı (`x0, z0, px_per_m`
haritası + oda noktaları `x_pct/y_pct`). Hepsi tek çerçevede.

Oynatıcı (`web/src/components/walk/`) **PlayCanvas 2.22** motoru: SplatTour'un da kullandığı,
SOG'un referans yükleyicisi. Önce `base`, sonra başlangıç odası, sonra sessizce diğerleri;
kameranın durduğu yere göre **kutu içerme** ile alan değiştirilir (en küçük içeren oda kazanır;
hiçbiri içermiyorsa base). Yürüyüş modu (WASD, 1.6 m göz hizası, kutuyla sınırlı), serbest uçuş,
tekerlekle görüş açısı, çift tıkla **oraya yürü** (`Picker.getWorldPointAsync` ile derinlik
okuması; duvara değil önündeki zemine durulur), kamera yolunu Catmull-Rom ile dolaşan **otomatik
tur**, tıklanabilir kat planı ve "buradasınız" işareti, oda rayı, dokunmatikte tek parmak
bak / iki parmak yürü / çift dokun. Zarf panoraması kamerayı izleyen büyük ters küredir.
Koordinat dönüşümü tek yerde (`walkMath.ts`, saf ve test edilmiş): motor çerçevesinden
oynatıcıya `(x, y, z) → (x, −y, −z)`.

**Teslimatta görüneni bozan dört şey** — ilk gerçek çekim hepsini aynı anda gösterdi; model
kusursuzken ekranda bulamaç vardı, ve dördü de eğitimde değil teslimattaydı:

- **Zarf küresi yanlış katmandaydı.** Kameranın etrafındaki 160 birimlik ters küre varsayılan
  dünya katmanındaydı; splatlar harmanlanan geçişte çizildiği için opak küre bütün sahnenin
  **üstünü** boyuyordu. Ekranda görülen şey odanın kendisi değil, o odanın 1024 px'lik bulanık
  panoramasıydı. Küre artık `LAYERID_SKYBOX`'ta: motor onu opak geometriden sonra, harmanlanan
  splatlardan önce çizer. Ayrıca `emissiveIntensity` 0.35: splat bulutu opak değildir (tipik
  bir karenin beşte biri saydamdır), yani küredeki her şey deliklere değil **her piksele**
  karışır.
- **Giriş pozu yürüyüşün ilk karesiydi.** Bir gezinti kapının önünde, telefon kapıya bir karış
  mesafede başlar; ziyaretçi oraya bırakılınca ekranı duvar kaplar. Artık her kare gördüğü
  noktaların medyan derinliğiyle oy verir ve **en derin bakış** kazanır; adaylar yürüyüşün
  durduğu yere (2–98 yüzdelik kutusu, medyan yüksekliğin ±0.6 m'si) sınırlıdır, yoksa haritacının
  dışarı fırlattığı bir kamera her seferinde kazanırdı — ölçüldü: zeminin 5–7 m altında/üstünde
  giriş pozları.
- **Alan seçimi en küçük kutuyu seçiyordu.** Ayrı çekilmiş odalar için doğru olan bu kural,
  parçalara bölünmüş **tek** bir yürüyüş için tamamen yanlıştır: parçalar aynı dairenin
  stretch'leridir, kutuları neredeyse tamamen örtüşür ve en küçük kutu her yerde kazanır.
  Ziyaretçi ana alanda dururken iki saniyelik bir parça çiziliyordu. Artık **kamera yolu en
  yakın geçen** alan kazanır; beraberlikte taban kazanır.
- **Temizlik fazla agresifti.** `min_opacity` 0.04, 3DGS'in alıştığı ~0.005'in sekiz katı;
  gerçek bir modelin %51'ini, duvarı dolduran soluk katmanlarla birlikte alıp götürüyordu.
  0.01'de kapsama 0.845 → 0.892. Rasterizer'ın splat yarıçapı tavanı da kareye göre ölçekleniyor
  (sabit 48 px, 640 px'lik bir küçük resimde yakın duvarı delik deşik ediyordu).

### Servis: iş kuyruğunda ikinci bir iş türü

`Job.kind` (`stitch` | `walkthrough`), `scene_id` artık boş olabilir (tur seviyesinde iş),
`Walkthrough` tablosu (tur başına bir; genel video, oda sahne id'leri, servis edilen
manifest), `Scene.walkthrough` (odanın 3B alanı; 360 ile yan yana). Aşama rayı iş türüne göre
(`WALK_STAGES`: upload, extract, features, sfm, register, train, clean, export, done), iş
ilerlemesi `StageTable` ile tür başına. SQLite'ta şema göçü elle: eksik kolonlar `ALTER TABLE`,
`job.scene_id`'nin boş olabilmesi tablo yeniden kurularak (`db._migrate_sqlite`).

Motor **kendi sürecinde** çalışır (`splat/worker.py`): pycolmap'in C++ çekirdeği de eğiticinin
GPU çalışma zamanı da bir iç CHECK'te süreci bitirir, ve bitirmemesi gereken süreç API'dir.
İş yöneticisi alt süreci başlatır, stdout'tan JSON ilerleme satırlarını okur, iptalde öldürür;
bir çökme ölü bir servis değil başarısız bir iştir.

Yükleme: bir oturumda genel video + oda videoları; küçükler için tek multipart (`spec` JSON +
`files[]`, dosyalar adlarıyla alanlara dağıtılır), telefon boyu için mevcut parça protokolü
(`PUT /properties/uploads/{id}/parts/{n}` → `POST /properties/{id}/walkthrough/from-parts`). Odalar
`capture_kind = "walk"` sahnelerdir; var olan bir oda `scene_id` ile adlandırılırsa 360'ı kalır,
3B alanı yanına gelir.

### Telefonla çekim: `/cek?m=w`

360 çekiminin sahnesi (`CaptureStage`) bir noktada **dönmeyi** dayatan bir makinedir
(istasyonlar, sıralar, kapsama halkaları, yürüyeni durduran dedektör); gezinti tam tersi
harekettir ve o makineye bir "mod" eklemek ikisini de bozardı. `WalkCaptureStage` ayrı bir
sahne, altındaki tesisatı paylaşır: `useCamera` (artık `lensPreference: "ultrawide"` — iOS
etiketinden "Ultra Wide" kamerayı, Android'de mantıksal kameranın zoom **tabanını** seçer;
beyaz dengesi kilidi), `useRecorder` (segmentler; `max_duration`'da biten segment
kaydı kesmez, bir sonrakini başlatır), `useMotion` (dönüş hızı 40°/s'yi 0.6 s aşarsa
"çok hızlı" uyarısı; solver bulanık kareyi eşleştiremez). Ekranlar SplatTour'un
rehberidir: kurallar → canlı önizleme + yürünen geçiş (odalar için dört geçiş, ilerleme
çubuğuyla; genel gezinti tek geçiş) → süre/boyut incelemesi → yükleme. `WalkCaptureFlow`
ziyareti yönetir: alan planı (genel + odalar), her alan **çekilir çekilmez** yüklenir
(`defer_build`: beş odalık bir dairenin videosu bir gigabaytı geçer ve telefon
tarayıcısı bunu blob'larda tutarak açık kalmaz), sonunda tek bir `rebuild` ile derleme
ve `JobProgress`. Masaüstünde aynı adres QR olur (`PhoneHandoff walk`); ev sayfasındaki
3B kartı oraya bağlanır.

### İkinci tur: ilk gerçek videodan ne öğrenildi (Ekim 2026)

Kullanıcının 82 saniyelik WhatsApp videosu (576×1024, 1 Mbps, bembeyaz boş daire) altı ayrı
parça olarak teslim edilmişti. Bu tur o videonun "neden altı parça" ve "neden iğneli" sorularını
ölçerek cevapladı; deney betikleri ve çıktıları `C:/biosphere-data/work/claude/walk/`.

**Parçalar videonun kendisinden geliyor, hattan değil.** Beş zincir kopuğu (11, 51, 60, 60,
72. saniyeler) ya kamera savrulurken bulanmış karelerdir ya da kadrajı dolduran çıplak
duvardır (51. saniyede altı ardışık karenin doku skoru 0,01–0,11; videonun medyanı 0,56).
Kullanıcı yürüyüşün %28'inde 40°/s'den hızlı döndü (çözülen karelerde medyan 22°/s, p95
98°/s; kopuklarda 90–220°/s). Üç yol denendi:

| yol | sonuç |
|---|---|
| bütün yürüyüşü kesmeden tek veritabanında artımlı haritalamak | 403 karenin 363'ü tek modelde (1,5–81,8 s), ama modelin içinde 8 imkânsız sıçrama: 33 ms arayla çekilmiş, 217 doğrulanmış eşleşmesi olan iki kare 10 birim (adımın 85 katı) uzağa konmuş — aynı mekânın iki kopyası. `split_at_jumps` modeli 9 parçaya böler; segmentlerden kötü |
| parçaları en büyüğüne 2D–3D yerelleştirmeyle (LightGlue + PnP) eklemek | 6 parçanın 0'ı: 7–17 kare yerelleşiyor ama ortak bir Sim3'e yalnızca 2–6'sı uyuyor (benzer odalar, beyaz duvar) |
| Depth Anything 3 ile kopukları köprülemek (Apache sürümleri) | berlin'de (yer gerçeği) ardışık kareler arası dönme hatası medyan 3,3–13° (gerçek hareket 8–15°), 150–180°'lik ters dönmeler; köprü için kullanılamaz |

Sonuç: bu video tek tutarlı bir model veremez; parçalı teslimat doğru yapıdır. Düzeltilen,
parçaların **yanlış yerleşmesi** ve oynatıcının parçaları **gösterememesi** oldu.

**Kayıt bekçileri (`register.py`, `pipeline.py`).** Teslim edilen turda "yerleşmiş" iki parça
30,8° ve 35,2° yatıktı (banyo eğik görünüyordu): kayıt artıkları küçüktü (2 cm), çünkü yanlış
çapa kareleri birbiriyle tutarlıydı. Artık bir oda ancak kameraları genel gezintininki gibi
dik tutulmuşsa yerleşir (ortalama yukarı ekseni tabanınkinden ≤ 12°; doğru kayıtlar 5–7°).
En küçük kareler ölçeği bir yazı-turaya bırakabiliyordu: bir parça −0,52 ölçekle (ayna
görüntüsü) "kaydolmuştu" — negatif ölçek artık hiç dönmez (eski hesap 20 rastgele denemenin
8'inde negatif veriyordu), ve paylaşılan kareler 0,3 m'den az yayılıyorsa ölçek gözlenemez
sayılır.

**Yoğun başlangıç bulutu (`densify.py`).** Eğitici SfM noktalarından başlar; beyaz duvarda
nokta yok, duvar büyüyen birkaç tohumla kaplanır — perdeler ve iğneler oradan. DA3-BASE'e
çözülmüş pozlar verilir, kare başına metrik derinlik döner (SfM noktalarıyla medyan %5,0
uyum; SMALL %6,9), komşu görüşlerin doğruladığı pikseller 2 cm vokselle inceltilip 50 bin
tohum olarak `init.ply`'ye yazılır. 79 karelik bir parçada, 8 ayrılmış karede: SfM başlangıcı
22,66 dB → DA3 23,09 dB (aynı SSIM 0,882), 221k yerine 423k splat; SfM'den daha çok büyütmek
(eşik yarıya) 22,08'e düşürür, yani fark başlangıçtan. DA3'ün büyük modelleri CC BY-NC'dir ve
kullanılmaz; kodu pip bağımlılığı değildir (opencv-python ve numpy<2 sabitler; contrib'i
siler), Brush gibi araç dizininden okunur. **Varsayılan kapalı** (`DenseInitConfig.enabled`):
yalnızca bir parçada ölçüldü, tam bir derlemede ve berlin bench'inde henüz değil.

**Temizlik (`clean.py`).** 3 m'lik dev tavanı aynı parçada teslimattan 0,71 dB alıyordu:
761 devin 716'sı kameralardan medyan 11 m uzaktaydı — pencereden görünen sokak. Tavan artık
yalnızca kamera yolunun 3 m yakınındaki devlere uygulanır (oda boyu lekeler; 45 splat, 0,04 dB).

**Oynatıcı.**
- *Kilitli parçalar.* Yerleşmemiş bir alan konumla asla seçilmez (`areaAt` bilerek dışarıda
  bırakır); raydan açıldığında da seçilmiyordu: ziyaretçi o alanın koordinatlarına
  ışınlanıyor, **tabanın** modeli çiziliyordu. Ada göre açılan yerleşmemiş alan artık
  kilitlenir; taban raya eklendi (dönüş yolu).
- *Rehberli tur telefonun baktığı yere bakar.* Eski tur yolun teğetine bakıyordu: teslim
  edilen tabanda bu, telefonun gerçek bakışından medyan 63° (p90 142°) sapma, zamanın %68'i
  45°'den fazla — modelin hiç görmediği yön. Manifestteki her kamera artık
  `[x, y, z, lx, ly, lz, t]` (bakış + videodaki saniye; API şeması kapalı olduğu için yeni
  anahtar değil, ilk üç sayıyı okuyan herkes için hâlâ bir kamera). Tur parçaları **çekim
  sırasıyla** dolaşır (giriş → odalar → kapı), yerleşmemiş parçaya kararmayla geçer.
- *Koridor.* Yürüme kutusu (yerleşmiş alanların kutularının birleşimi) teslim edilen turda
  161 m² zemin açıyordu; %38'i telefonun durduğu yere 1 m'den yakındı, %41'i 2 m'den uzak.
  Yürüyüş modu artık kamera yoluna 1 m'den uzaklaşamaz (sınırda kayar).
- *Açılış görüş açısı.* Dikey telefonun yatay alanı ~63°; 16:9 ekranda 70° dikey, 102°
  yatay demektir ve karenin yarısı telefonun o noktadan hiç bakmadığı şeydir. Açılışta yatay
  alan 80° ile sınırlanır (yakınlaştırma/uzaklaştırma ziyaretçinin).
- *Yakın kırpma reddedildi.* Kameranın 0,3 / 0,5 / 0,8 m önünü kırpmak ayrılmış karelerde
  −0,09 / −0,25 / −0,87 dB: yakındaki splatlar gerçek duvarlar.

**Çekim (`WalkCaptureStage`).** Çıplak duvar uyarısı: önizlemenin 96 px'lik gri küçük
resminde 8×8 hücrelerin ayrıntılı (gri seviye sapması > 6) oranı; ALIKED nokta sayısıyla 0,67
korelasyonlu. Temiz kurulan berlin'de karelerin %1,8'i, gerçek yürüyüşte %2,0'ı 0,15'in
altında — yani sürekli bir durum değil bir an; 0,5 s sürerse uyarı çıkar. 51. saniyedeki
kopuk tam böyle bir andır. Teslimat da dönüşü ölçer (`sfm.turn_speeds`, çözülmüş ardışık
kareler): berlin karelerinin %2,5–8,7'si 40°/s'nin üstünde (p90 33°/s), parçalanan gerçek
yürüyüşün %28'i (p90 57°/s); %15'i aşan yürüyüş manifestte `turned_too_fast` uyarısı alır.

**Hız.** Zincir sondası her ardışık çift için LightGlue'yu baştan kuruyordu (dokuz katmanı
CPU'da rastgele başlatıp ağırlıkları üstüne yüklemek); kalabalık bir dizüstünde 82 saniyelik
bir yürüyüşün sondası 25 dakikayı `kaiming_uniform_` içinde, GPU boştayken geçirdi
(py-spy). Ağlar süreç başına bir kez kurulur; `match_pairs` yalnızca çiftlerin andığı
kareleri GPU'ya taşır.

### Bugün yapamadıkları

- **Metrik ölçek varsayımdır.** 1.45 m kamera yüksekliği; çömelerek çekilen bir oda küçük,
  kolu yukarıda çekilen büyük çıkar. Derinlik modeli için yer ayrıldı, takılmadı.
- **Pozlama telafisi yok.** Otomatik pozlamayla çekilen bir video pencerenin yanında parlak,
  köşede karanlık splatlar üretir; AE kilidi çekim talimatının bir parçası.
- **Tek kat.** `floor_index` taşınıyor, kat rayı yok; her kat ayrı bir tur olarak yüklenir.
- **Çarpışma yok.** Kutu dışına çıkılmaz ama duvarın içinden geçilebilir; SplatTour'un
  "yürünebilirlik maskesi"nin karşılığı yazılmadı.
- **Kapı eşiğinde çift geometri.** Oda değişimi sert bir kutu geçişidir; iki ayrı splat aynı
  kapıyı iki kez çizer ve geçiş anında biri ötekine "atlar". SplatTour'da da böyle.

---

## 16. İlan metni: evden sahibinden ilanına

Tur bitti; sırada ilanın kendisi var. Emlakçı bir başlık ve bir paragraf yazacak — birisi bu
metni okuyup daireyi görmeye gelip gelmeyeceğine karar verecek. Hedef biçim belirlidir:
**sahibinden.com'un "İlan Başlığı" ve "Açıklama" kutularına olduğu gibi yapıştırılabilecek**
bir metin. O metnin malzemesi zaten sistemde: oda adları, kat planından gelen **ölçülmüş**
taban alanları, tavan yüksekliği ve odaların kendi görüntüleri. `biosphere/ai/listing.py`
bunları toplar, resimlere bakabilen bir modele verir ve geri gelen metni **kendi verdiği
verilere karşı denetler**.

```
odalar + layout.json (ölçüm) ─► FACTS bloğu ─┐
odanın initial_view'ından 768 px pencere ────┼─► model ─► başlık + açıklama + öne çıkanlar
(panorama yoksa odanın ilk fotoğrafı)        │              │
                                             │              ▼
                                             └────────► denetim: her m², her fiyat,
                                                        her "metroya 5 dakika"
```

### Evin iç adı modele hiç gitmiyor

İlk gerçek koşuda model başlığı **"7eylul — camlı kapılı, beyaz tonlarda kompakt oda"** yazdı
ve açıklamayı "7eylul, beyaz yüzeyleri ve sade hatlarıyla…" diye açtı. `7eylul` emlakçının
panelde eve verdiği dosya adıdır — bir tarih, bir kod, bir dizüstü bilgisayar markası — ve
daire hakkında bir bilgi değildir. Prompt'a "bunu kullanma" yazmak yetmez: bağlamdaki bir ad,
metnin uzanabileceği bir addır. Çözüm onu **hiç göndermemek** oldu; `_facts_block` evin
başlığını taşımaz, `compose_offline` da metni ondan kurmaz, ve `digest` artık başlığı
içermez — çünkü ad artık bir girdi değilse, adı değiştirmek yazılmış metni çöpe atmamalı.

### Emlakçının notu bir kaynaktır, denetlenen taraf değildir

Bir panorama katı, cepheyi, ısıtmayı, asansörü, "3+1"i ölçemez — ama dairede duran insan
bunları bilir. Panelin "Ek not" kutusu bu yüzden vardır ve prompt'ta **yetkili bir olgu
kaynağı** olarak tanımlıdır: notta yazan şey emlakçının kendi mülkü hakkındaki beyanıdır,
model onu kullanır. Bunun sonucu `audit()`'e de yansır: notta geçen bir alan ya da terim
uyarı üretmez. Aksi hâlde emlakçı "asansörlü, 8. kat" yazacak ve panel ona kendi cümlesini
"doğrulanamaz iddia" diye geri gösterecekti — bir uyarı panelini görmezden gelmeyi öğreten
tam olarak budur. **Portal kuralları muaf değildir**: telefon, e-posta ve link, kim istemiş
olursa olsun uyarı üretir, çünkü sahibinden o ilanı yayınlamaz.

### sahibinden'in kendi kuralları

Kaynak: [İlan Verme Kuralları](https://www.sahibinden.com/sozlesmeler/ilan-verme-kurallari-48)
ve [İlan Verme İpuçları](https://yardim.sahibinden.com/hc/tr/articles/115004672353).
Açıklama ve başlıkta telefon numarası, e-posta, web adresi, link ya da firma iletişim bilgisi
yasak — numara ilanın kendi kutusunda yayınlanır. Prompt bunları yasaklar, `audit()` üçünü de
(`contact_phone`, `contact_email`, `contact_link`) metinde arar. Başlık için resmî bir karakter
sınırı yayınlanmıyor; vitrinde başlığın yalnızca ilk birkaç kelimesi okunduğu için kural
"**ne olduğu ve en güçlü somut özelliği ilk kelimelerde**, 50–65 karakter" biçiminde yazıldı.
Başlığın virgülle ayrılmış bir özellik listesi ("Daire, yaklaşık 12,8 m² oda ve yüksek tavan")
değil, tek bir doğal isim tamlaması olması ("Camlı bölmeli, yaklaşık 12,8 m² odalı daire") ve
metnin "detaylı bilgi için iletişime geçin" ile bitmemesi de aynı yerde yazılıdır — ikisi de
ilk koşularda çıkan gerçek kusurlardı.

### Üç karar

**Modele bilgi sorulmuyor, bilgi veriliyor.** Alanlar burada, kat planının okuduğu aynı
`layout.json` dosyalarından hesaplanır ve prompt'a bir FACTS bloğu olarak girer. "Bu oda kaç
metrekare" diye sorulan bir görme modeli akıcı ve yanlış cevap verir; "bu oda 24,3 m², bunun
hakkında yaz" denen modelin tahmin edecek bir şeyi kalmaz. Toplam da verilir ve ne olduğu
açıkça yazılır: **ölçülen odaların toplamı**, dairenin alanı değil — ölçülmemiş odalar ve
kimsenin içinde durmadığı koridorlar bu sayının dışındadır.

**Sonra cevap denetlenir.** `audit()` bitmiş metindeki her m² değerini geri okur ve verilmemiş
bir sayıyı tanımaz; para birimi işaretlerini, "tapu/kredi/aidat", "metroya beş dakika" gibi
hiçbir fotoğrafın taşıyamayacağı iddiaları arar. Bulduğu her şey yanıtın `warnings` alanına
yazılır ve panelde **metnin üstünde**, yayına alacak kişinin göreceği yerde durur. Metne
dokunulmaz: bir cümleyi sessizce silmek, geri kalanının denetlendiği izlenimini bırakır ki bu
kendi başına bir yalandır. Oda boşaltmadaki argümanın aynısı — uydurma bir gardırop, yumuşak
bir duvardan kötüdür.

**Modelsiz de cevap verir.** Anahtar yoksa, istemci kütüphanesi kurulu değilse ya da servise
ulaşılamazsa `compose_offline()` aynı verilerden düz ve doğru bir paragraf yazar; yanıt
`model_backed: false` der ve nedeni `warnings`'te durur. İyi bir metin değildir — doğru
kutuların içinde doğru sayılarla duran bir başlangıçtır, yani bu özellik yokken elde olan şey.
Butonun 500 döndürmesi, kartı dolmuş bir kullanıcı için daha kötü bir üründür.

### Resimler

Oda başına bir **dik izdüşüm penceresi** (768 px, 95°), sahnenin kendi `initial_view`
yönünden — emlakçının odanın "önü" olarak seçtiği yön. Equirect'i olduğu gibi göndermek
nezaket meselesi değil: 2:1 tuvalde model bükülmüş bir oda ve karenin üçte birine yayılmış bir
kapı görür, ve o bükülmeyi tarif eder. Panoraması olmayan ama fotoğrafı olan oda ilk
fotoğrafını gönderir; ikisi de yoksa oda yalnızca FACTS'te adıyla geçer, çünkü görülmemiş bir
oda tarif edilemez. Varsayılan tavan altı resimdir (`BIOSPHERE_LISTING_MAX_IMAGES`): kapak
odası önce, sonra tur sırası.

### Servis

```
GET  /api/v1/properties/{id}/listing   panel açılışı. Diskteki metni okur, model çağırmaz.
POST /api/v1/properties/{id}/listing   yazar. Saniyeler, bir ağ gidiş-dönüşü ve para.
```

Metin `<tour_dir>/listing.json` içinde, FACTS'in parmak iziyle birlikte saklanır: oda eklemek,
oda adını değiştirmek, bir odayı ölçmek ya da öbür dilde sormak önbelleği ıskalar, geri kalan
her şey ısabet eder. `refresh` bunun tek istisnasıdır — "yanlış değil ama başka bir cümle
istiyorum" gerçek bir istektir. **Kaydetmek bu uçta değildir**: üretilen metin bir formdaki
öneridir ve eve yazılması `PATCH /properties/{id}`'dir, başlığın hep gittiği yer. Bu uca bir
`apply` bayrağı eklemek, `tour.title` üzerine ikinci bir yazıcı koymak olurdu.

Model yalnızca bu özellik için ağa çıkar ve motorun binadan dışarı çıkan tek parçasıdır; bu
yüzden `BIOSPHERE_LISTING_API_KEY` (boşsa `ANTHROPIC_API_KEY`) tanımlanana kadar kapalıdır ve
istemci ayrı bir extra'dır (`pip install -e ".[listing]"`), `ai` extra'sının içinde değil:
diskteki ağırlıklarla, başkasının servisine giden bir istemci aynı türden bağımlılık değildir.

### Ne kadara mal oluyor

Her üretim, harcadığı token'ı ve dolar karşılığını yanıtın `usage` alanında geri getirir ve
panel bunu metnin yanında gösterir — "bu bize ne kadara mal oluyor" sorusu cevaba bakarken
sorulur, bir hafta sonra sunucu günlüğünde değil. Fiyat `llm.PRICES` tablosundan gelir
(1M token başına USD; opus-5 5/25, sonnet-5 2/10, haiku-4.5 1/5) ve tabloda olmayan bir model
`null` döner: komşu modelin fiyatından tahmin edilmiş bir rakam, bu özelliği çalıştırıp
çalıştırmamaya karar veren birine gösterilecek en kötü şeydir. Sürüm ekli anlık adlar
(`claude-opus-5-20260114`) önekten fiyatlanır.

Aynı ev (tek oda, tek resim), üç model, aynı prompt:

| model | girdi | çıktı | süre | maliyet |
|-------|-------|-------|------|---------|
| `claude-opus-5` | 3.342 | 2.186 | 34 s | $0,0714 |
| `claude-sonnet-5` | 3.881 | 5.063 | 52 s | $0,0584 |
| **`claude-haiku-4-5`** (varsayılan) | 3.121 | 441 | 14 s | **$0,0053** |

Dört odalı, üç resimli bir ev opus'ta $0,080 ettiği yerde haiku'da bir sentin altında kalır.
**Varsayılan haiku**, çünkü bu buton çekilen her evde basılıyor ve ondalık burada gerçek bir
sayıya dönüşüyor: yüz ilan opus'ta yedi dolar, haiku'da elli sent.

Ölçümdeki sürpriz `sonnet-5`: opus'a yakın bir fatura çıkardı, çünkü girdi değil **çıktı**
patladı (5.063 token — aynı metin için haiku'nun on bir katı). Bu iş çıktı ağırlıklı olmadığı
için "orta segment model" burada bir tasarruf değil; seçim pratikte haiku ile opus arasındadır.

Ucuz modelin verdiği şey **düzyazı kalitesidir, doğruluk değil**: sayılar yine FACTS
bloğundan gelir ve `audit()` yine hepsini denetler. Ölçülen fark üslupta: haiku 55–90 kelime
arasında gezinir, yer yer kitabi Türkçe kurar ("Pencere panosuyla karşısında"), ve iki koşuda
ölçülen 2,59 m'yi başlıkta "yüksek tavan" diye sattı — o yüzden `ceiling_not_high` uyarısı
prompt'a değil **denetime** yazıldı. Cümlenin kendisi işin merkezindeyse
`BIOSPHERE_LISTING_MODEL=claude-opus-5` bir satırlık değişikliktir.

Girdinin büyük bölümü resimlerdir; `BIOSPHERE_LISTING_MAX_IMAGES` onu doğrudan keser. Önbellek
yüzünden paneli ikinci açış her hâlükârda bedavadır.

---

## 17. VR: turu gözlükle gezmek

Hem 360 tur hem 3B gezinti iki yoldan VR'a girer, ve hangisinin açılacağına cihaz karar verir,
kullanıcı değil (`components/vr/capabilities.ts`, `chooseVrRoute`):

| Ortam | Yol |
|---|---|
| `navigator.xr` `immersive-vr` destekliyor (Quest, Vision Pro, uygun Android) | **WebXR** |
| Gözlük tarayıcısı ama bir portalın iframe'i XR'ı engellemiş | **Yeni sekmede aç** (aynı oda) |
| Telefon, yön sensörü var | **VR gözlük (Cardboard)** — yan yana stereo |
| Telefon, ama iframe tam ekranı/sensörü vermiyor ya da iPhone | **Yeni sekmede aç** |
| Masaüstü | Düğme hiç görünmez |

**Panorama tek göze çizilir ve doğrusu budur.** Bir equirect panorama sonsuzdaki bir küredir; iki
gözün ona bakışı arasında paralaks yoktur, sahte bir derinlik eklemek yanlış olurdu. 3B gezinti
ise gerçek bir sahnedir ve iki göz IPD kadar ayrık render edilir.

**Odalar arası geçiş VR'da kararma ile olur, kapıya yürüyüş ile değil.** 2B oynatıcının
`PortalTransit` dolly'si başın hareket etmediği bir kamera hareketidir — VR'da mide bulandıran
şeyin ta kendisi. Halkaya bakıp beklemek (1,5 sn, Google'ın Cardboard yönergesindeki süre;
altında insanlar sadece göz attıkları şeyleri seçmeye başlar) ya da kumanda tetiği seçer; ekran
kararır, oda değişir, açılır.

**Cardboard lens modeli ölçülmüş değil, yayımlanmış ya da tahmin edilmiştir.** Cardboard v2 için
Google'ın değerleri (k1 0,34, k2 0,55, 64 mm mercek arası); piyasadaki "VR Box" türü gözlükler için
tahmin (k1 0,28, k2 0,24, 62 mm) — varsayılan budur, ölçülmüş bir gözlükle değiştirilene kadar.
Sahada `?vrlens=k1,k2,ipdMm,ekranMm,fovDeg` ile ayarlanır; rehber ekranında iki profil
arasında seçim vardır. iPhone 14'te göz başına doku 1098×980 (VR Box) / 1374×1176 (Cardboard).

**Gezintide ışınlanma yalnızca çekimin gördüğü koridora iner** (`teleport.ts`, 2B yürüyüşle aynı
`walkMath` koridoru): splat bulutu kameranın hiç durmadığı yerden bakılınca çöptür ve VR'da
oraya gitmek, gördüğün her şeyin çözülmesi demektir. Kehribar halka "oraya gidilmez" der.
Quest'te gezinti WebGL2 ile çalışır, çünkü PlayCanvas WebGPU'dan XR sunamaz (splat sıralaması o
yüzden CPU'da).

**Gömme.** Portal iframe'i `allow="fullscreen; xr-spatial-tracking; accelerometer; gyroscope;
magnetometer; screen-wake-lock"` taşımazsa tarayıcı XR'ı ve sensörleri reddeder; panonun verdiği
gömme kodu ve açılış sayfasındaki örnek bunu taşır, `/embed` yanıtı da aynı izinleri
`Permissions-Policy` ile verir. Taşımayan bir portalda düğme "yeni sekmede aç"a döner.

**Neyin doğrulandığı.** Saf matematik (lens, yön, bakış isabeti, bekleme sayacı, ışınlanma)
104 birim testiyle; Cardboard görünümü masaüstünde `?vr=cardboard` ile (fare sürükleme baş
hareketinin yerine geçer) ekranda. Gerçek bir gözlükte ya da telefonda **hiçbiri** henüz
denenmedi: XR oturumu, kumandalar, iOS izin akışı, Quest'te kare hızı. Paket maliyeti: 360
ana paketi +3,8 KB gzip; VR sahnesi ve gezinti VR çalışma zamanı ayrı, tembel yüklenen
parçalardır (11,4 KB ve 11,9 KB).
