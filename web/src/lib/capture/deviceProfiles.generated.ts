/**
 * GENERATED FILE -- DO NOT EDIT.
 *
 * Written by engine/scripts/gen_device_profiles.py from
 * engine/biosphere/assets/device_profiles.json, which is the single source of truth for
 * what lens each phone has. The engine bounds its focal search with these numbers; the
 * capture app uses them to coach before recording. If you edit this file by hand the two
 * will disagree, the app will coach for one lens while the engine solves for another, and
 * nothing will warn you.
 *
 * `equivMm` is the 35mm-equivalent focal length on the DIAGONAL convention (CIPA, 43.267
 * mm), for the 4:3 frame the app records. See biosphere/pano/devices.py for the two unit
 * traps that make this wrong if you take a number from anywhere else.
 */

export interface PhoneLens {
  id: string;
  brand: string;
  model: string;
  tier: "flagship" | "upper-mid" | "mid" | "budget";
  equivMm: number;
  ois: boolean;
  lidar: boolean;
  /** Exact `navigator.userAgentData` model strings, lower-cased. Empty for every iPhone. */
  aliases: string[];
}

/** One iPhone screen, and the models that share it. Several, usually. */
export interface IosFingerprint {
  cssWidth: number;
  cssHeight: number;
  dpr: number;
  equivMm: number;
  /** "yes" only when every model behind this screen has the scanner. */
  lidar: "yes" | "no" | "maybe";
  models: string[];
}

export const PHONE_LENSES: readonly PhoneLens[] = [
  {
    "id": "iphone-11",
    "brand": "Apple",
    "model": "iPhone 11",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-11-pro",
    "brand": "Apple",
    "model": "iPhone 11 Pro",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-11-pro-max",
    "brand": "Apple",
    "model": "iPhone 11 Pro Max",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-12",
    "brand": "Apple",
    "model": "iPhone 12",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-12-pro",
    "brand": "Apple",
    "model": "iPhone 12 Pro",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-12-pro-max",
    "brand": "Apple",
    "model": "iPhone 12 Pro Max",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-12-mini",
    "brand": "Apple",
    "model": "iPhone 12 mini",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-13",
    "brand": "Apple",
    "model": "iPhone 13",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-13-pro",
    "brand": "Apple",
    "model": "iPhone 13 Pro",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-13-pro-max",
    "brand": "Apple",
    "model": "iPhone 13 Pro Max",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-13-mini",
    "brand": "Apple",
    "model": "iPhone 13 mini",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-14",
    "brand": "Apple",
    "model": "iPhone 14",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-14-plus",
    "brand": "Apple",
    "model": "iPhone 14 Plus",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-14-pro",
    "brand": "Apple",
    "model": "iPhone 14 Pro",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-14-pro-max",
    "brand": "Apple",
    "model": "iPhone 14 Pro Max",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-15",
    "brand": "Apple",
    "model": "iPhone 15",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-15-plus",
    "brand": "Apple",
    "model": "iPhone 15 Plus",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-15-pro",
    "brand": "Apple",
    "model": "iPhone 15 Pro",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-15-pro-max",
    "brand": "Apple",
    "model": "iPhone 15 Pro Max",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-16",
    "brand": "Apple",
    "model": "iPhone 16",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-16-plus",
    "brand": "Apple",
    "model": "iPhone 16 Plus",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-16-pro",
    "brand": "Apple",
    "model": "iPhone 16 Pro",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-16-pro-max",
    "brand": "Apple",
    "model": "iPhone 16 Pro Max",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-16e",
    "brand": "Apple",
    "model": "iPhone 16e",
    "tier": "upper-mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-17",
    "brand": "Apple",
    "model": "iPhone 17",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-17-pro",
    "brand": "Apple",
    "model": "iPhone 17 Pro",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-17-pro-max",
    "brand": "Apple",
    "model": "iPhone 17 Pro Max",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": true,
    "aliases": []
  },
  {
    "id": "iphone-air",
    "brand": "Apple",
    "model": "iPhone Air",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "iphone-se-2022",
    "brand": "Apple",
    "model": "iPhone SE (2nd/3rd gen)",
    "tier": "budget",
    "equivMm": 28.0,
    "ois": true,
    "lidar": false,
    "aliases": []
  },
  {
    "id": "pixel-10",
    "brand": "Google",
    "model": "Pixel 10",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 10"
    ]
  },
  {
    "id": "pixel-10-pro",
    "brand": "Google",
    "model": "Pixel 10 Pro",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 10 pro",
      "pixel 10 pro xl"
    ]
  },
  {
    "id": "pixel-6",
    "brand": "Google",
    "model": "Pixel 6",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 6"
    ]
  },
  {
    "id": "pixel-6a",
    "brand": "Google",
    "model": "Pixel 6a",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 6a"
    ]
  },
  {
    "id": "pixel-7",
    "brand": "Google",
    "model": "Pixel 7",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 7"
    ]
  },
  {
    "id": "pixel-7a",
    "brand": "Google",
    "model": "Pixel 7a",
    "tier": "mid",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 7a"
    ]
  },
  {
    "id": "pixel-8",
    "brand": "Google",
    "model": "Pixel 8",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 8"
    ]
  },
  {
    "id": "pixel-8a",
    "brand": "Google",
    "model": "Pixel 8a",
    "tier": "mid",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 8a"
    ]
  },
  {
    "id": "pixel-9",
    "brand": "Google",
    "model": "Pixel 9",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 9"
    ]
  },
  {
    "id": "pixel-9-pro",
    "brand": "Google",
    "model": "Pixel 9 Pro",
    "tier": "flagship",
    "equivMm": 25.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "pixel 9 pro",
      "pixel 9 pro xl"
    ]
  },
  {
    "id": "honor-90",
    "brand": "Honor",
    "model": "Honor 90",
    "tier": "upper-mid",
    "equivMm": 24.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "rea-nx9"
    ]
  },
  {
    "id": "honor-x9b",
    "brand": "Honor",
    "model": "Honor X9b",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "ali-nx1",
      "ali-n21"
    ]
  },
  {
    "id": "huawei-nova-12s",
    "brand": "Huawei",
    "model": "Huawei nova 12s",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "foa-lx9",
      "foa-nx9"
    ]
  },
  {
    "id": "oppo-a78",
    "brand": "Oppo",
    "model": "Oppo A78 4G",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "cph2565",
      "cph2495"
    ]
  },
  {
    "id": "oppo-reno-12",
    "brand": "Oppo",
    "model": "Oppo Reno12",
    "tier": "upper-mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "cph2637",
      "cph2625"
    ]
  },
  {
    "id": "realme-12-pro",
    "brand": "Realme",
    "model": "Realme 12 Pro 5G",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "rmx3842",
      "rmx3841"
    ]
  },
  {
    "id": "realme-c55",
    "brand": "Realme",
    "model": "Realme C55",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "rmx3710",
      "rmx3711"
    ]
  },
  {
    "id": "galaxy-a15",
    "brand": "Samsung",
    "model": "Galaxy A15",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "sm-a155f",
      "sm-a155f/ds",
      "sm-a155m",
      "sm-a156b"
    ]
  },
  {
    "id": "galaxy-a34",
    "brand": "Samsung",
    "model": "Galaxy A34 5G",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-a346b",
      "sm-a346b/ds",
      "sm-a346e",
      "sm-a346m"
    ]
  },
  {
    "id": "galaxy-a54",
    "brand": "Samsung",
    "model": "Galaxy A54 5G",
    "tier": "upper-mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-a546b",
      "sm-a546b/ds",
      "sm-a546e",
      "sm-a546u"
    ]
  },
  {
    "id": "galaxy-a55",
    "brand": "Samsung",
    "model": "Galaxy A55 5G",
    "tier": "upper-mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-a556b",
      "sm-a556b/ds",
      "sm-a556e"
    ]
  },
  {
    "id": "galaxy-a56",
    "brand": "Samsung",
    "model": "Galaxy A56 5G",
    "tier": "upper-mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-a566b",
      "sm-a566b/ds",
      "sm-a566e"
    ]
  },
  {
    "id": "galaxy-s21",
    "brand": "Samsung",
    "model": "Galaxy S21 5G",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-g991b",
      "sm-g991b/ds",
      "sm-g991u",
      "sm-g991n"
    ]
  },
  {
    "id": "galaxy-s21-ultra",
    "brand": "Samsung",
    "model": "Galaxy S21 Ultra 5G",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-g998b",
      "sm-g998b/ds"
    ]
  },
  {
    "id": "galaxy-s21-plus",
    "brand": "Samsung",
    "model": "Galaxy S21+ 5G",
    "tier": "flagship",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-g996b",
      "sm-g996b/ds"
    ]
  },
  {
    "id": "galaxy-s22",
    "brand": "Samsung",
    "model": "Galaxy S22 5G",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s901b",
      "sm-s901b/ds",
      "sm-s901u"
    ]
  },
  {
    "id": "galaxy-s22-ultra",
    "brand": "Samsung",
    "model": "Galaxy S22 Ultra",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s908b",
      "sm-s908b/ds"
    ]
  },
  {
    "id": "galaxy-s23",
    "brand": "Samsung",
    "model": "Galaxy S23",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s911b",
      "sm-s911b/ds"
    ]
  },
  {
    "id": "galaxy-s23-ultra",
    "brand": "Samsung",
    "model": "Galaxy S23 Ultra",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s918b",
      "sm-s918b/ds"
    ]
  },
  {
    "id": "galaxy-s24",
    "brand": "Samsung",
    "model": "Galaxy S24",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s921b",
      "sm-s921b/ds"
    ]
  },
  {
    "id": "galaxy-s24-ultra",
    "brand": "Samsung",
    "model": "Galaxy S24 Ultra",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s928b",
      "sm-s928b/ds"
    ]
  },
  {
    "id": "galaxy-s25",
    "brand": "Samsung",
    "model": "Galaxy S25",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s931b",
      "sm-s931b/ds"
    ]
  },
  {
    "id": "galaxy-s25-ultra",
    "brand": "Samsung",
    "model": "Galaxy S25 Ultra",
    "tier": "flagship",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s938b",
      "sm-s938b/ds"
    ]
  },
  {
    "id": "galaxy-s26-ultra",
    "brand": "Samsung",
    "model": "Galaxy S26 Ultra",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "sm-s948b"
    ]
  },
  {
    "id": "redmi-note-11",
    "brand": "Xiaomi",
    "model": "Redmi Note 11",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "2201117tg",
      "2201117ty",
      "2201117sg"
    ]
  },
  {
    "id": "redmi-note-12",
    "brand": "Xiaomi",
    "model": "Redmi Note 12 4G",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "23021raaeg",
      "23027rad4g",
      "22111317g"
    ]
  },
  {
    "id": "redmi-note-13",
    "brand": "Xiaomi",
    "model": "Redmi Note 13 4G",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "23129raa4g",
      "2312crad3c"
    ]
  },
  {
    "id": "redmi-note-13-pro",
    "brand": "Xiaomi",
    "model": "Redmi Note 13 Pro 4G",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "2312dra50g",
      "2312dra50c"
    ]
  },
  {
    "id": "redmi-note-14",
    "brand": "Xiaomi",
    "model": "Redmi Note 14 4G",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "24117rk2cg",
      "2411drn47g"
    ]
  },
  {
    "id": "xiaomi-13",
    "brand": "Xiaomi",
    "model": "Xiaomi 13",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "2211133g",
      "2211133c"
    ]
  },
  {
    "id": "xiaomi-14",
    "brand": "Xiaomi",
    "model": "Xiaomi 14",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "23127pn0cg",
      "23127pn0cc"
    ]
  },
  {
    "id": "xiaomi-15",
    "brand": "Xiaomi",
    "model": "Xiaomi 15",
    "tier": "flagship",
    "equivMm": 23.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "24129pn74g",
      "24129pn74c"
    ]
  },
  {
    "id": "vivo-v29",
    "brand": "vivo",
    "model": "vivo V29 5G",
    "tier": "upper-mid",
    "equivMm": 24.0,
    "ois": true,
    "lidar": false,
    "aliases": [
      "v2250",
      "v2250a"
    ]
  },
  {
    "id": "vivo-v40-lite",
    "brand": "vivo",
    "model": "vivo V40 Lite",
    "tier": "mid",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "v2354",
      "v2404"
    ]
  },
  {
    "id": "vivo-y17s",
    "brand": "vivo",
    "model": "vivo Y17s",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "v2310"
    ]
  },
  {
    "id": "vivo-y22",
    "brand": "vivo",
    "model": "vivo Y22",
    "tier": "budget",
    "equivMm": 26.8,
    "ois": false,
    "lidar": false,
    "aliases": [
      "v2207"
    ]
  },
  {
    "id": "vivo-y22s",
    "brand": "vivo",
    "model": "vivo Y22s",
    "tier": "budget",
    "equivMm": 26.8,
    "ois": false,
    "lidar": false,
    "aliases": [
      "v2206"
    ]
  },
  {
    "id": "vivo-y36",
    "brand": "vivo",
    "model": "vivo Y36",
    "tier": "budget",
    "equivMm": 26.0,
    "ois": false,
    "lidar": false,
    "aliases": [
      "v2247",
      "v2248"
    ]
  }
] as const;

export const IOS_FINGERPRINTS: readonly IosFingerprint[] = [
  {
    "cssWidth": 375,
    "cssHeight": 667,
    "dpr": 2.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone 6",
      "iPhone 6s",
      "iPhone 7",
      "iPhone 8",
      "iPhone SE (2nd gen)",
      "iPhone SE (3rd gen)"
    ]
  },
  {
    "cssWidth": 414,
    "cssHeight": 736,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone 6 Plus",
      "iPhone 6s Plus",
      "iPhone 7 Plus",
      "iPhone 8 Plus"
    ]
  },
  {
    "cssWidth": 375,
    "cssHeight": 812,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone X",
      "iPhone XS",
      "iPhone 11 Pro",
      "iPhone 12 mini",
      "iPhone 13 mini"
    ]
  },
  {
    "cssWidth": 414,
    "cssHeight": 896,
    "dpr": 2.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone XR",
      "iPhone 11"
    ]
  },
  {
    "cssWidth": 414,
    "cssHeight": 896,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone XS Max",
      "iPhone 11 Pro Max"
    ]
  },
  {
    "cssWidth": 390,
    "cssHeight": 844,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "maybe",
    "models": [
      "iPhone 12",
      "iPhone 12 Pro",
      "iPhone 13",
      "iPhone 13 Pro",
      "iPhone 14",
      "iPhone 16e"
    ]
  },
  {
    "cssWidth": 428,
    "cssHeight": 926,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "maybe",
    "models": [
      "iPhone 12 Pro Max",
      "iPhone 13 Pro Max",
      "iPhone 14 Plus"
    ]
  },
  {
    "cssWidth": 393,
    "cssHeight": 852,
    "dpr": 3.0,
    "equivMm": 25.0,
    "lidar": "maybe",
    "models": [
      "iPhone 14 Pro",
      "iPhone 15",
      "iPhone 15 Pro",
      "iPhone 16"
    ]
  },
  {
    "cssWidth": 430,
    "cssHeight": 932,
    "dpr": 3.0,
    "equivMm": 25.0,
    "lidar": "maybe",
    "models": [
      "iPhone 14 Pro Max",
      "iPhone 15 Plus",
      "iPhone 15 Pro Max",
      "iPhone 16 Plus"
    ]
  },
  {
    "cssWidth": 402,
    "cssHeight": 874,
    "dpr": 3.0,
    "equivMm": 25.0,
    "lidar": "maybe",
    "models": [
      "iPhone 16 Pro",
      "iPhone 17",
      "iPhone 17 Pro"
    ]
  },
  {
    "cssWidth": 440,
    "cssHeight": 956,
    "dpr": 3.0,
    "equivMm": 24.0,
    "lidar": "yes",
    "models": [
      "iPhone 16 Pro Max",
      "iPhone 17 Pro Max"
    ]
  },
  {
    "cssWidth": 420,
    "cssHeight": 912,
    "dpr": 3.0,
    "equivMm": 26.0,
    "lidar": "no",
    "models": [
      "iPhone Air"
    ]
  }
] as const;
