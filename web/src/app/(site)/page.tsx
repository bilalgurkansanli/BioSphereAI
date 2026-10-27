import { Contribute } from "@/components/site/Contribute";
import { Gallery } from "@/components/site/Gallery";
import { Hero } from "@/components/site/Hero";
import { Highlights } from "@/components/site/Highlights";
import { Pipeline } from "@/components/site/Pipeline";
import { QuickStart } from "@/components/site/QuickStart";
import { Status } from "@/components/site/Status";
import { Story } from "@/components/site/Story";

/**
 * The open-source project's front page, in the order a curious engineer needs it.
 *
 * 1. What it is, in one line, beside the player itself (a synthetic room, dragged by hand).
 * 2. What it looks like once installed — screenshots from a clean, synthetic-only install.
 * 3. What works and what does not. This is the core of the page: a status table that only had
 *    green rows would be a brochure, so the parked work is stated in the developer's own words.
 * 4. Why it is open source, honestly and without naming anybody else.
 * 5. How it works, measured numbers, how to run it, and where help would matter.
 *
 * Nothing is sold here. There is no sign-up call to action, no pricing and no company; the
 * dashboard's own sign-in exists for whoever runs an installation, and the header links to it
 * quietly. Title, description and the Open Graph image come from the root layout and
 * `opengraph-image.jpg` beside this file (rendered from the synthetic room's poster).
 */

export default function LandingPage() {
  return (
    <div className="relative isolate overflow-x-clip">
      <Hero />
      <Gallery />
      <Status />
      <Story />
      <Pipeline />
      <Highlights />
      <QuickStart />
      <Contribute />
    </div>
  );
}
