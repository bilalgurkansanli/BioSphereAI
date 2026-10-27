"use client";

/**
 * /rehber — running BioSphere on your own machine, step by step.
 *
 * Nothing here is hosted: there is no account to open and no server of ours to send anything
 * to. Everything this page describes happens on the reader's own computer, so it is written
 * as a path they can follow top to bottom, with every command copyable.
 */

import { AlertTriangle, CheckCircle2, Cpu, Globe2, KeyRound, Smartphone, Terminal, Wrench } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { CLONE_URL, REPO_NAME, REPO_URL, SITE_ONLY } from "@/lib/project";
import type { Text } from "./copy";
import { CodeBlock, GitHubMark, Section, useText } from "./primitives";

interface Step {
  title: Text;
  body: Text;
  code?: { tr: string; en: string } | string;
  note?: Text;
}

const STEPS: Step[] = [
  {
    title: { tr: "Gerekenleri kurun", en: "Install the prerequisites" },
    body: {
      tr: "Git, Python paket yöneticisi uv, Node.js ve FFmpeg. Windows'ta hepsi winget ile tek satırda kurulur; her birini kurduktan sonra terminali kapatıp yeniden açın.",
      en: "Git, the uv Python manager, Node.js and FFmpeg. On Windows each is one winget line; close and reopen the terminal after installing.",
    },
    code: `winget install Git.Git
winget install astral-sh.uv
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg`,
    note: {
      tr: "NVIDIA ekran kartınız varsa güncel sürücüyü de kurun; 3B gezinti ve yapay zekâ araçları GPU ister. 360 tur için GPU gerekmez.",
      en: "If you have an NVIDIA card, install a current driver too; the 3D walkthrough and the AI tools need a GPU. The 360 tour does not.",
    },
  },
  {
    title: { tr: "Projeyi indirin", en: "Get the code" },
    body: {
      tr: "Depoyu istediğiniz bir klasöre klonlayın.",
      en: "Clone the repository into any folder.",
    },
    code: `git clone ${CLONE_URL}
cd ${REPO_NAME}`,
  },
  {
    title: { tr: "Kurulumu çalıştırın", en: "Run the setup" },
    body: {
      tr: "Betik Python ortamını ve web bağımlılıklarını kurar, sonunda makinenizi kontrol eder. Windows yerel betikleri varsayılan olarak çalıştırmadığı için komut bu izni yalnızca bu çalıştırma için verir.",
      en: "The script creates the Python environment, installs the web dependencies and checks your machine at the end. Windows blocks local scripts by default, so the command allows it for this one run.",
    },
    code: {
      tr: `powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1
# GPU'nuz varsa (yapay zekâ araçları için):
powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1 -Gpu`,
      en: `powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1
# with a GPU (for the AI tools):
powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1 -Gpu`,
    },
  },
  {
    title: { tr: "Başlatın", en: "Start it" },
    body: {
      tr: "scripts klasöründeki start-dev.cmd dosyasına çift tıklayın. Motor ve panel kendi pencerelerinde açılır; durdurmak için pencereyi kapatmanız yeterli.",
      en: "Double-click start-dev.cmd in the scripts folder. The engine and the dashboard open in their own windows; close a window to stop it.",
    },
    code: `scripts\\start-dev.cmd`,
    note: {
      tr: "Motor açıldıktan sonraki ilk dakikada panel GPU'yu ve araçları denetlerken \"kontrol ediliyor\" diyebilir; birkaç saniye bekleyin.",
      en: "In the first minute after the engine starts, the dashboard may say \"checking\" while it probes the GPU and tools; give it a few seconds.",
    },
  },
  {
    title: { tr: "Paneli açın", en: "Open the dashboard" },
    body: {
      tr: "Tarayıcıda http://localhost:3000/evler adresine gidin. Giriş ekranı yok: kendi bilgisayarınızda çalıştığı için panel doğrudan açılır.",
      en: "Go to http://localhost:3000/evler. There is no sign-in: it runs on your own machine, so the dashboard opens straight away.",
    },
    code: "http://localhost:3000/evler",
  },
  {
    title: { tr: "İlk turunuzu oluşturun", en: "Make your first tour" },
    body: {
      tr: "\"Yeni ev\" ile bir ev oluşturun, \"Sahne ekle\" → \"Video yükle\" ile odanın ortasında telefonu yavaşça kendi etrafında çevirdiğiniz bir videoyu yükleyin. Yaklaşık üç dakikada panorama hazır olur; tur sayfasından oynatıcıyı açıp diğer odaları ekleyebilirsiniz.",
      en: "Create a property with \"Yeni ev\", then \"Sahne ekle\" → \"Video yükle\" a video where you slowly turned the phone around in the middle of a room. The panorama is ready in about three minutes; open the player from the property page and add the other rooms.",
    },
    note: {
      tr: "İyi bir çekim için: telefonu dikey tutun, kendi ekseninizde yavaşça dönün; göz hizasında bir tur, sonra biraz yukarı ve biraz aşağı bakarak birer tur. Videoyu mesajlaşma uygulamasından değil, telefonun kendisinden aktarın.",
      en: "For a good capture: hold the phone upright, turn slowly on the spot; one turn at eye level, then one looking slightly up and one slightly down. Copy the video off the phone itself, not through a messenger.",
    },
  },
];

interface Extra {
  icon: ReactNode;
  title: Text;
  body: Text;
  code?: string;
}

const EXTRAS: Extra[] = [
  {
    icon: <Smartphone className="size-[18px]" strokeWidth={1.7} aria-hidden />,
    title: { tr: "Telefonla rehberli çekim", en: "Guided capture on a phone" },
    body: {
      tr: "Telefon kamerası yalnızca HTTPS üzerinde açılır. Bilgisayarınızın yerel ağ adresini scripts/certs/san.cnf dosyasına IP.2 olarak ekleyin, bir kez sertifika üretin; start-dev.cmd telefon vekilini de açar. Telefonda https://<bilgisayarın-IP'si>:3443 adresini açıp sertifika uyarısını bir kez kabul edin, sonra \"Sahne ekle\" → \"Telefonla çek\" QR kodunu okutun.",
      en: "A phone camera only opens over HTTPS. Add your computer's LAN address to scripts/certs/san.cnf as IP.2 and generate a certificate once; start-dev.cmd then starts the phone proxy too. On the phone open https://<your-LAN-IP>:3443, accept the certificate warning once, then scan the QR under \"Sahne ekle\" → \"Telefonla çek\".",
    },
    code: "openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout scripts/certs/key.pem -out scripts/certs/cert.pem -config scripts/certs/san.cnf -extensions v3",
  },
  {
    icon: <KeyRound className="size-[18px]" strokeWidth={1.7} aria-hidden />,
    title: { tr: "İlan metni ve yatırım asistanı", en: "Listing copy and the investment assistant" },
    body: {
      tr: "Kendi Anthropic API anahtarınızla çalışır; proje anahtarla gelmez. Eki bir kez kurun, sonra panelde Ayarlar → \"Anthropic API anahtarı\" alanına anahtarınızı yapıştırıp \"Anahtarı test et\"e basın. Motoru yeniden başlatmanız gerekmez. Anahtar yokken bu özellikler kapalı kalır; geri kalan her şey çalışır.",
      en: "Runs on your own Anthropic API key; the project ships without one. Install the extra once, then paste your key under Settings → \"Anthropic API key\" in the dashboard and press \"Test the key\". No engine restart needed. Without a key these features stay off; everything else works.",
    },
    code: `cd engine
uv pip install -e ".[listing]"
# then: http://localhost:3000/ayarlar`,
  },
  {
    icon: <Cpu className="size-[18px]" strokeWidth={1.7} aria-hidden />,
    title: { tr: "Mobilya kaldırma ve dekor (GPU)", en: "Furniture removal and staging (GPU)" },
    body: {
      tr: "Kurulumu -Gpu ile çalıştırın ve modellerin ilk kullanımda inebilmesi için engine/.env içine izni ekleyin. Bu araçlar prototip aşamasında; model lisanslarını THIRD_PARTY_NOTICES dosyasında görebilirsiniz.",
      en: "Run the setup with -Gpu and let the models download on first use. These tools are prototypes; the model licences are listed in THIRD_PARTY_NOTICES.",
    },
    code: "echo BIOSPHERE_AI_ALLOW_DOWNLOADS=true >> engine\\.env",
  },
  {
    icon: <Globe2 className="size-[18px]" strokeWidth={1.7} aria-hidden />,
    title: { tr: "3B gezinti (GPU)", en: "3D walkthrough (GPU)" },
    body: {
      tr: "Eki kurun, Brush v0.3.0 ikilisini ve @playcanvas/splat-transform paketini indirip yollarını engine/.env içinde verin (örnek değerler .env.example'da). Panelde ev sayfasındaki 3B kartı yürüyüş videosunu yükleyip modeli eğitir; GPU ile yaklaşık yarım saat sürer.",
      en: "Install the extra, download the Brush v0.3.0 binary and @playcanvas/splat-transform and point engine/.env at them (examples in .env.example). The 3D card on the property page uploads a walk-through video and trains the model; about half an hour on a GPU.",
    },
    code: `cd engine
uv pip install -e ".[walkthrough]"`,
  },
];

const LINUX = `# Linux / macOS
git clone ${CLONE_URL} && cd ${REPO_NAME}
cd engine && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m uvicorn biosphere.api.main:app --port 8008 &
cd ../web && npm install && npm run dev`;

const TROUBLE: { q: Text; a: Text }[] = [
  {
    q: { tr: "\"Betik çalıştırma bu sistemde devre dışı\" hatası", en: "\"Running scripts is disabled on this system\"" },
    a: {
      tr: "Komutu powershell -ExecutionPolicy Bypass -File ile çalıştırın ya da scripts\\start-dev.cmd kullanın; .cmd dosyaları bu ayardan etkilenmez.",
      en: "Run it with powershell -ExecutionPolicy Bypass -File, or use scripts\\start-dev.cmd; .cmd files are not affected by that setting.",
    },
  },
  {
    q: { tr: "Port kullanımda (8008 ya da 3000)", en: "Port already in use (8008 or 3000)" },
    a: {
      tr: "Önceki pencereler hâlâ açık olabilir; onları kapatıp yeniden başlatın.",
      en: "Earlier windows may still be open; close them and start again.",
    },
  },
  {
    q: { tr: "Telefonda sayfa açılıyor ama dokunmalara tepki vermiyor", en: "The page opens on the phone but nothing responds" },
    a: {
      tr: "Telefonu bilgisayarla aynı ağa bağlayın ve adresi IP ile açın (https://<IP>:3443); geliştirme sunucusu yalnızca yerel ağ adreslerine izin verir.",
      en: "Put the phone on the same network and open the address by IP (https://<IP>:3443); the dev server only allows local network addresses.",
    },
  },
  {
    q: { tr: "iPhone sertifika yüzünden sayfayı açmıyor", en: "The iPhone refuses the page because of the certificate" },
    a: {
      tr: "Sertifikadaki IP, telefonda yazdığınız adresle aynı olmalı. san.cnf'ye doğru IP'yi ekleyip sertifikayı yeniden üretin.",
      en: "The certificate must contain the exact IP you typed on the phone. Add it to san.cnf and generate the certificate again.",
    },
  },
  {
    q: { tr: "Bir özellik görünmüyor ya da kapalı", en: "A feature is missing or disabled" },
    a: {
      tr: "http://127.0.0.1:8008/api/v1/health adresi makinenizde neyin çalıştığını söyler: FFmpeg, GPU, 3B gezinti araçları ve yapay zekâ modelleri.",
      en: "http://127.0.0.1:8008/api/v1/health tells you what works on your machine: FFmpeg, GPU, the 3D walkthrough tools and the AI models.",
    },
  },
];

const T = {
  eyebrow: { tr: "Rehber", en: "Guide" },
  title: { tr: "BioSphere'i kendi bilgisayarınızda çalıştırın", en: "Run BioSphere on your own computer" },
  lead: {
    tr: "BioSphere bir hizmet değil, indirip çalıştırdığınız bir yazılım: her şey sizin bilgisayarınızda olur, hiçbir veri bize gelmez. Windows'ta yaklaşık on beş dakikalık kurulumdan sonra ilk turunuzu oluşturabilirsiniz.",
    en: "BioSphere is not a service but software you download and run: everything happens on your machine and no data comes to us. After about fifteen minutes of setup on Windows you can make your first tour.",
  },
  steps: { tr: "Adım adım (Windows)", en: "Step by step (Windows)" },
  extras: { tr: "İsteğe bağlı özellikler", en: "Optional features" },
  linux: { tr: "Linux ve macOS", en: "Linux and macOS" },
  linuxBody: {
    tr: "Betikler Windows için yazıldı; diğer sistemlerde aynı adımları elle çalıştırın.",
    en: "The scripts target Windows; on other systems run the same steps by hand.",
  },
  panelTitle: { tr: "Panel bu sitede değil, sizin bilgisayarınızda", en: "The dashboard runs on your computer, not on this site" },
  panelBody: {
    tr: "Bu sayfa projenin tanıtımı. Videoyu 360 tura çeviren motor, paneliniz, odalarınız ve API anahtarınız kendi bilgisayarınızda çalışır; buraya hiçbir şey yüklenmez. Aşağıdaki adımlardan sonra panel http://localhost:3000/evler adresinde açılır.",
    en: "This page presents the project. The engine that turns a video into a 360 tour, your dashboard, your rooms and your API key all run on your own computer; nothing is uploaded here. After the steps below the dashboard opens at http://localhost:3000/evler.",
  },
  trouble: { tr: "Sorun giderme", en: "Troubleshooting" },
  publicTitle: { tr: "Başkalarının da erişeceği bir kurulum mu?", en: "Running it for other people?" },
  publicBody: {
    tr: "Panel varsayılan olarak giriş istemez, çünkü kendi bilgisayarınızda yalnızca size açıktır. Başkalarının ulaşabileceği bir sunucuda çalıştıracaksanız engine/.env içinde hesap modunu açın, HTTPS kullanın ve kendi gizlilik metninizi yayımlayın.",
    en: "The dashboard asks for no sign-in by default, because on your own machine only you can reach it. If you run it on a server other people can reach, switch on accounts mode in engine/.env, use HTTPS and publish your own privacy notice.",
  },
  done: { tr: "Takıldığınız bir yer olursa GitHub'da bir issue açın.", en: "Stuck somewhere? Open an issue on GitHub." },
  issue: { tr: "Issue aç", en: "Open an issue" },
} satisfies Record<string, Text>;

export function Guide() {
  const { tx, locale } = useText();
  const codeOf = (code: Step["code"]) => (code === undefined ? null : typeof code === "string" ? code : code[locale]);

  return (
    <>
      <Section className="pt-16 sm:pt-20">
        <p className="font-mono text-[11.5px] font-medium tracking-[0.14em] text-bio uppercase">{tx(T.eyebrow)}</p>
        <h1 className="mt-3 max-w-3xl text-[clamp(2rem,4.6vw,3.1rem)] leading-[1.05] font-semibold tracking-[-0.035em] text-ink">
          {tx(T.title)}
        </h1>
        <p className="mt-5 max-w-2xl text-[15.5px] leading-relaxed text-ink-muted">{tx(T.lead)}</p>
        {SITE_ONLY ? (
          <div id="panel" className="mt-8 max-w-2xl scroll-mt-24 rounded-lg border border-bio/25 bg-bio/5 p-4 sm:p-5">
            <p className="text-[14px] font-semibold text-ink">{tx(T.panelTitle)}</p>
            <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-muted">{tx(T.panelBody)}</p>
          </div>
        ) : null}
      </Section>

      <Section aria-labelledby="rehber-adimlar" className="mt-14">
        <h2 id="rehber-adimlar" className="flex items-center gap-2 text-[17px] font-semibold tracking-tight text-ink">
          <Terminal className="size-4 text-bio" aria-hidden /> {tx(T.steps)}
        </h2>
        <ol className="mt-6 space-y-4">
          {STEPS.map((step, index) => (
            <li key={step.title.en} className="grid grid-cols-[auto_minmax(0,1fr)] gap-4 rounded-lg border border-line bg-surface/40 p-5 sm:gap-5 sm:p-6">
              <span className="grid size-9 place-items-center rounded-full border border-bio/40 bg-bio/10 font-mono text-[13px] font-semibold text-bio">
                {index + 1}
              </span>
              <div className="min-w-0">
                <h3 className="text-[16px] font-semibold tracking-tight text-ink">{tx(step.title)}</h3>
                <p className="mt-1.5 max-w-3xl text-[14px] leading-relaxed text-ink-muted">{tx(step.body)}</p>
                {codeOf(step.code) ? <CodeBlock code={codeOf(step.code) as string} className="mt-3.5" /> : null}
                {step.note ? (
                  <p className="mt-3 flex max-w-3xl gap-2 text-[12.5px] leading-relaxed text-ink-faint">
                    <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-bio" aria-hidden />
                    <span>{tx(step.note)}</span>
                  </p>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
      </Section>

      <Section aria-labelledby="rehber-ekler" className="mt-16">
        <h2 id="rehber-ekler" className="flex items-center gap-2 text-[17px] font-semibold tracking-tight text-ink">
          <Wrench className="size-4 text-bio" aria-hidden /> {tx(T.extras)}
        </h2>
        <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {EXTRAS.map((extra) => (
            <article key={extra.title.en} className="flex min-w-0 flex-col rounded-lg border border-line bg-surface/40 p-5 sm:p-6">
              <div className="flex items-center gap-3">
                <span className="grid size-9 place-items-center rounded-sm border border-line bg-surface-2 text-ink-muted">{extra.icon}</span>
                <h3 className="text-[15px] font-semibold tracking-tight text-ink">{tx(extra.title)}</h3>
              </div>
              <p className="mt-3 text-[13.5px] leading-relaxed text-ink-muted">{tx(extra.body)}</p>
              {extra.code ? <CodeBlock code={extra.code} className="mt-auto pt-3.5" /> : null}
            </article>
          ))}
        </div>
      </Section>

      <Section aria-labelledby="rehber-linux" className="mt-16">
        <h2 id="rehber-linux" className="text-[17px] font-semibold tracking-tight text-ink">{tx(T.linux)}</h2>
        <p className="mt-3 max-w-2xl text-[14px] leading-relaxed text-ink-muted">{tx(T.linuxBody)}</p>
        <CodeBlock code={LINUX} className="mt-4" />
      </Section>

      <Section aria-labelledby="rehber-sorun" className="mt-16">
        <h2 id="rehber-sorun" className="flex items-center gap-2 text-[17px] font-semibold tracking-tight text-ink">
          <AlertTriangle className="size-4 text-amber" aria-hidden /> {tx(T.trouble)}
        </h2>
        <dl className="mt-6 divide-y divide-line overflow-hidden rounded-lg border border-line">
          {TROUBLE.map((item) => (
            <div key={item.q.en} className="grid grid-cols-1 gap-1.5 p-5 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] sm:gap-6">
              <dt className="text-[14px] font-semibold text-ink">{tx(item.q)}</dt>
              <dd className="text-[13.5px] leading-relaxed text-ink-muted">{tx(item.a)}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section className="mt-16 pb-24">
        <div className="rounded-lg border border-line bg-surface/40 p-5 sm:p-6">
          <h2 className="text-[15px] font-semibold tracking-tight text-ink">{tx(T.publicTitle)}</h2>
          <p className="mt-2 max-w-3xl text-[13.5px] leading-relaxed text-ink-muted">{tx(T.publicBody)}</p>
          <CodeBlock code={"BIOSPHERE_AUTH_MODE=accounts\nBIOSPHERE_ENV=production"} className="mt-3.5" />
        </div>
        <p className="mt-8 flex flex-wrap items-center gap-3 text-[14px] text-ink-muted">
          {tx(T.done)}
          <Link
            href={`${REPO_URL}/issues`}
            className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-[13px] font-medium text-ink transition-colors hover:border-line-strong"
          >
            <GitHubMark className="size-3.5" /> {tx(T.issue)}
          </Link>
        </p>
      </Section>
    </>
  );
}
