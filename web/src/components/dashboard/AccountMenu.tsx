"use client";

import { LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useSession } from "@/components/auth/SessionProvider";
import { DropdownMenu } from "@/components/ui/DropdownMenu";
import { ui } from "./copy";
import { useLocale } from "@/lib/hooks/useLocale";

/** Initials from a display name, for the avatar when there is no picture. */
function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toLocaleUpperCase("tr-TR");
  return (parts[0][0] + parts[parts.length - 1][0]).toLocaleUpperCase("tr-TR");
}

/**
 * Who is signed in, and the way out.
 *
 * The e-mail address is shown next to the name because an agency with two accounts on one
 * machine is the normal case for this product, and "which of us is this?" is a question the
 * avatar alone cannot answer.
 */
export function AccountMenu() {
  const { locale } = useLocale();
  const { user, signOut } = useSession();
  const router = useRouter();

  // Local mode (the open-source default): no account was signed into, so there is no menu.
  if (!user || user.local) return null;

  const leave = async () => {
    try {
      await signOut();
    } finally {
      // Whatever the engine said, this browser is signed out — go somewhere that makes
      // sense for a signed-out visitor rather than leaving them on a guarded page.
      router.replace("/giris");
      toast.success(ui("auth.signOut", locale));
    }
  };

  return (
    <DropdownMenu
      label={ui("auth.account", locale)}
      header={
        <>
          <span className="block font-medium text-ink">{user.name}</span>
          <span className="block break-all text-ink-faint">{user.email}</span>
        </>
      }
      items={[
        {
          id: "signout",
          label: ui("auth.signOut", locale),
          icon: LogOut,
          onSelect: () => void leave(),
        },
      ]}
      trigger={
        <span
          className="grid size-8 place-items-center rounded-full border border-line bg-surface-2 text-[11px] font-semibold text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
          title={`${user.name} · ${user.email}`}
        >
          {initials(user.name)}
        </span>
      }
    />
  );
}
