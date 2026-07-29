#!/data/data/com.termux/files/usr/bin/bash
# claude-code-android migration: pinned v2.x  ->  current auto-updating architecture
#
# For existing users on the old pinned Path A install (npm package
# @anthropic-ai/claude-code, typically 2.1.112, locked read-only with the
# in-process auto-updater disabled). This moves you to the current auto-updating
# architecture (patched native linux-arm64 binary + auto-updating wrapper)
# WITHOUT losing your work.
#
# Preserved untouched: your chats/sessions, OAuth login, settings.json,
# and any custom agents/hooks/skills/CLAUDE.md under ~/.claude.
#
# Safety:
#   - A full backup of ~/.claude, ~/.claude.json, and ~/.bashrc is taken
#     BEFORE anything destructive, with a restore.sh you can run to undo.
#   - The new binary is downloaded, checksum-verified, and patched BEFORE the
#     old install is removed, so a failure leaves your old install usable.
#   - Run this only when NO claude session is active.
#
# Fresh installs should use install.sh instead, not this script.
#
# SYNC NOTE: the npm-version resolve, download, checksum, patchelf, and the
# emitted wrapper below are kept byte-identical to install.sh. If you change
# one, change both. This is enforced, not just documented: the sections are
# wrapped in matching "# SYNC:BEGIN <name>" / "# SYNC:END <name>" markers, and
# scripts/check-sync.sh diffs them and fails if they drift (wired into CI).
#
# Tracking the upstream issue this works around:
#   https://github.com/anthropics/claude-code/issues/50270

set -euo pipefail

info(){ printf '\033[0;36m[info]\033[0m  %s\n' "$1"; }
ok(){   printf '\033[0;32m[ok]\033[0m    %s\n' "$1"; }
warn(){ printf '\033[0;33m[warn]\033[0m  %s\n' "$1" >&2; }
fail(){ printf '\033[0;31m[fail]\033[0m  %s\n' "$1" >&2; exit 1; }

# DNS ETIMEOUT fix: preload sets Bun's c-ares resolver to a live nameserver (the wrapper loads it via BUN_OPTIONS).
CC_SETDNS="$HOME/.local/share/claude/setdns.js"
CC_SETDNS_JS='try { require("dns").setServers(["8.8.8.8", "8.8.4.4"]); } catch (e) {}'
write_setdns() {
  [ -s "$1" ] && return 0
  printf '%s\n' "$CC_SETDNS_JS" > "$1" 2>/dev/null
}

BACKUP_DIR=""
on_err(){
  ec=$?
  printf '\033[0;31m[fail]\033[0m  migration stopped (exit %s).\n' "$ec" >&2
  if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
    printf '        Your data backup is at: %s\n' "$BACKUP_DIR" >&2
    printf '        Restore it with:        bash %s/restore.sh\n' "$BACKUP_DIR" >&2
  fi
  exit "$ec"
}
trap on_err ERR

# --- Preflight ---
if [ -z "${PREFIX:-}" ]; then
  fail "PREFIX unset. Run this inside native Termux, not adb shell or proot."
fi
if [ "$PREFIX" != "/data/data/com.termux/files/usr" ]; then
  fail "This runs in native Termux only (PREFIX is $PREFIX). Path B/C installs do not need it."
fi
if [ "$(uname -m)" != "aarch64" ]; then
  fail "aarch64 only. uname -m reports: $(uname -m)"
fi
# Validate HOME before any path operation derives from it. Every destructive
# target must rest on a verified base path, never an assumption.
if [ -z "${HOME:-}" ] || [ ! -d "$HOME" ]; then
  fail "HOME is unset or not a directory; refusing to run."
fi

# No live claude session: replacing the binary under a running session corrupts it.
RUNNING="$( { pgrep -x claude; pgrep -f '@anthropic-ai/claude-code'; } 2>/dev/null | sort -un | grep -vw "$$" | grep -vw "${PPID:-0}" | tr '\n' ' ' || true )"
if [ -n "${RUNNING// /}" ]; then
  fail "claude appears to be running (PIDs: $RUNNING). Close all claude sessions, then re-run."
fi

# A live proot session: the package-upgrade step can update proot packages under it.
if pgrep -x proot >/dev/null 2>&1; then
  warn "A proot session is running. The package-upgrade step can update proot packages underneath it."
  read -r -p "Continue anyway? [y/N] " PR
  case "${PR,,}" in
    y|yes) ;;
    *) fail "Aborted. Close the proot session and re-run." ;;
  esac
fi

cat <<'BANNER'

  claude-code-android migration  (pinned v2.x  ->  current architecture)
  ======================================================================

BANNER

# --- Detect current install ---
NPM_PKG="$PREFIX/lib/node_modules/@anthropic-ai/claude-code"
BINLINK="$PREFIX/bin/claude"
VERSIONS_DIR="$HOME/.local/share/claude/versions"

state="foreign"
if [ -d "$VERSIONS_DIR" ] && ls "$VERSIONS_DIR"/*.*.* >/dev/null 2>&1 && [ -f "$BINLINK" ] && [ ! -L "$BINLINK" ]; then
  state="already_v29"
elif [ -d "$NPM_PKG" ]; then
  state="pinned"
elif [ -L "$BINLINK" ] && readlink "$BINLINK" | grep -q 'node_modules/@anthropic-ai/claude-code'; then
  state="pinned"
elif [ -d "$VERSIONS_DIR" ] && ls "$VERSIONS_DIR"/*.*.* >/dev/null 2>&1 && [ ! -e "$BINLINK" ]; then
  # Official native install: a versioned binary under ~/.local/share/claude with
  # a ~/.local/bin launcher, but no $PREFIX/bin wrapper and no npm package.
  # claude treats it as native; convert it to the wrapper in place, keeping data.
  state="native"
elif [ ! -e "$BINLINK" ] && [ ! -d "$NPM_PKG" ] && ! command -v claude >/dev/null 2>&1; then
  state="fresh"
fi

case "$state" in
  already_v29)
    ok "You are already on the v2.9 architecture (wrapper + versioned binary)."
    info "The wrapper auto-updates. Force a check with: claude --update-now"
    info "To refresh the launcher itself (to recover from a crashing update or to"
    info "pick up launcher improvements), re-run install.sh; migrate.sh is only for"
    info "moving an older pinned npm install onto this architecture."
    trap - ERR
    exit 0
    ;;

  fresh)
    fail "No existing claude install found. This is the upgrade path; for a fresh install run install.sh."
    ;;
  foreign)
    warn "Found a 'claude' this migrator did not install:"
    if [ -e "$BINLINK" ]; then ls -l "$BINLINK" >&2; fi
    if command -v claude >/dev/null 2>&1; then warn "claude resolves to: $(command -v claude)"; fi
    fail "Refusing to touch an install I did not create. Remove it yourself then run install.sh, or open an issue."
    ;;
esac

OLD_VER="$(claude --version 2>&1 | head -1 || echo unknown)"
if [ "$state" = native ]; then
  ok "Detected an official native install (reports: $OLD_VER). Converting it in place."
else
  ok "Detected pinned v2.x install (reports: $OLD_VER)."
fi
echo

# --- Q: recommended packages ---
cat <<'Q'
Install recommended packages (git, gh, wget, jq, python, openssh, tree, proot,
termux-api, proot-distro, make, clang, file, xxd, htop, bat, fzf)? Already
installed ones are skipped. Choose no if you manage these yourself.
Q
read -r -p "Install recommended packages? [Y/n] " QR
QR="${QR:-Y}"
case "${QR,,}" in
  y|yes) RECOMMENDED=1 ;;
  n|no)  RECOMMENDED=0 ;;
  *) fail "answer 'y' or 'n'; got '$QR'" ;;
esac
echo

# --- Migration summary + explicit go ---
cat <<'SUMMARY'
This will:
  1. Back up ~/.claude, ~/.claude.json, and ~/.bashrc to a timestamped folder.
  2. Download, verify, and patch the latest claude linux-arm64 binary.
  3. Replace the old claude binary (the npm package is removed only if present).
  4. Install the auto-updating wrapper.
  5. Merge your settings.json, preserving your existing hooks/permissions/env.

Preserved untouched: your chats/sessions, login, agents, hooks, skills, CLAUDE.md.
SUMMARY
read -r -p "Proceed? [y/N] " GO
case "${GO,,}" in
  y|yes) ;;
  *) fail "Aborted by user. Nothing changed." ;;
esac
echo

# --- Backup (data first, before anything destructive) ---
STAMP="$(date -u +%Y%m%d-%H%M%S)"
BACKUP_DIR="$HOME/claude-migration-backup-$STAMP"
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
info "backing up to $BACKUP_DIR"
if [ -e "$HOME/.claude" ]; then
  # No -h: preserve symlinks as symlinks (e.g. config symlinked into a repo),
  # so a restore re-creates the links rather than duplicating their targets.
  tar czf "$BACKUP_DIR/dot-claude.tgz" -C "$HOME" .claude || fail "backup of ~/.claude failed"
fi
if [ -e "$HOME/.claude.json" ]; then cp -a "$HOME/.claude.json" "$BACKUP_DIR/"; fi
if [ -e "$HOME/.bashrc" ]; then cp -a "$HOME/.bashrc" "$BACKUP_DIR/"; fi
{
  echo "pre-version: $OLD_VER"
  echo "bin: $(ls -l "$BINLINK" 2>&1)"
  echo "node: $(node -v 2>&1 || echo none)"
  echo "date_utc: $STAMP"
} > "$BACKUP_DIR/pre-state.txt"

cat > "$BACKUP_DIR/restore.sh" <<'RESTORE'
#!/data/data/com.termux/files/usr/bin/bash
# Restore the data captured before this migration.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
echo "Restoring ~/.claude, ~/.claude.json, ~/.bashrc from $here ..."
if [ -f "$here/dot-claude.tgz" ]; then
  # Verify the archive is readable BEFORE removing the live directory, so a
  # corrupt backup can never leave you with neither the old nor the backup.
  tar tzf "$here/dot-claude.tgz" >/dev/null 2>&1 || { echo "backup archive is unreadable; aborting restore to avoid data loss."; exit 1; }
  rm -rf "$HOME/.claude"
  tar xzf "$here/dot-claude.tgz" -C "$HOME"
fi
if [ -f "$here/.claude.json" ]; then cp -a "$here/.claude.json" "$HOME/.claude.json"; fi
if [ -f "$here/.bashrc" ]; then cp -a "$here/.bashrc" "$HOME/.bashrc"; fi
echo "Data restored. Your sessions and login are back regardless of which binary you run."
echo "To reinstall the old pinned binary (optional):"
echo "  npm install -g @anthropic-ai/claude-code@2.1.112"
RESTORE
chmod +x "$BACKUP_DIR/restore.sh"
ok "backup complete (restore: bash $BACKUP_DIR/restore.sh)"

# --- apt options: existing user, preserve their configs ---
export DEBIAN_FRONTEND=noninteractive
APT_OPTS="-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"

# Pin a Termux mirror if none is selected (only-if-missing; never overrides a
# working mirror), so the package step cannot stall on a mirror-selection prompt.
if [ ! -e "$PREFIX/etc/termux/chosen_mirrors" ] && [ -e "$PREFIX/etc/termux/mirrors/default" ]; then
  ln -sf "$PREFIX/etc/termux/mirrors/default" "$PREFIX/etc/termux/chosen_mirrors" 2>/dev/null || true
fi

# apt-get (not pkg/apt) for the scripted steps: apt-get has a stable CLI and
# does not print apt's "does not have a stable CLI interface" script warning.
info "apt-get update"
apt-get update $APT_OPTS >/dev/null || fail "apt-get update failed"
info "apt-get full-upgrade"
apt-get full-upgrade $APT_OPTS >/dev/null || fail "apt-get full-upgrade failed"
info "apt-get install curl jq"
apt-get install $APT_OPTS curl jq >/dev/null || fail "apt-get install curl/jq failed"

info "apt-get install glibc-repo"
apt-get install $APT_OPTS glibc-repo >/dev/null || fail "glibc-repo install failed"
apt-get update $APT_OPTS >/dev/null || fail "apt-get update after glibc-repo failed"
info "apt-get install glibc-runner patchelf-glibc (~50 MB)"
apt-get install $APT_OPTS glibc-runner patchelf-glibc >/dev/null || fail "glibc-runner install failed"

PATCHELF="$PREFIX/glibc/bin/patchelf"
GLIBC_LD="$PREFIX/glibc/lib/ld-linux-aarch64.so.1"
if [ ! -x "$PATCHELF" ]; then fail "patchelf not found at $PATCHELF"; fi
if [ ! -f "$GLIBC_LD" ]; then fail "glibc ld.so not found at $GLIBC_LD"; fi
ok "glibc-runner + patchelf ready"

# --- Resolve + stage the NEW binary (old install still intact at this point) ---
# SYNC:BEGIN resolve-download-patch (kept byte-identical to install.sh: checked by scripts/check-sync.sh)
info "resolving latest claude version from npm registry"
LATEST="$(curl -fsSL --max-time 10 https://registry.npmjs.org/@anthropic-ai/claude-code/latest 2>/dev/null | jq -r .version 2>/dev/null)"
if [ -z "$LATEST" ] || [ "$LATEST" = "null" ]; then
  fail "could not query npm registry for the latest claude version"
fi
if ! printf '%s' "$LATEST" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  fail "npm registry returned an unexpected version string: $LATEST"
fi
ok "latest claude version: $LATEST"

VERSIONS_DIR="$HOME/.local/share/claude/versions"
BINARY="$VERSIONS_DIR/$LATEST"
WRAPPER="$PREFIX/bin/claude"
mkdir -p "$VERSIONS_DIR" "$HOME/.claude"

DL_BASE="https://downloads.claude.ai/claude-code-releases/$LATEST"

info "downloading $LATEST linux-arm64 binary (~233 MB)"
curl -fsSL --max-time 300 "$DL_BASE/linux-arm64/claude" -o "$BINARY.tmp" \
  || { rm -f "$BINARY.tmp"; fail "binary download failed"; }

info "verifying checksum against published manifest"
EXP="$(curl -fsSL --max-time 10 "$DL_BASE/manifest.json" 2>/dev/null | jq -er '.platforms["linux-arm64"].checksum' 2>/dev/null || true)"
ACT="$(sha256sum "$BINARY.tmp" | cut -d' ' -f1)"
if [ -z "$EXP" ]; then
  rm -f "$BINARY.tmp"
  fail "could not read checksum from manifest"
fi
if [ "$EXP" != "$ACT" ]; then
  rm -f "$BINARY.tmp"
  fail "checksum mismatch: expected $EXP, got $ACT"
fi
ok "checksum verified"

chmod +x "$BINARY.tmp"
LD_PRELOAD='' "$PATCHELF" --set-interpreter "$GLIBC_LD" "$BINARY.tmp" \
  || { rm -f "$BINARY.tmp"; fail "patchelf failed to set ELF interpreter"; }
mv "$BINARY.tmp" "$BINARY"
# SYNC:END resolve-download-patch
ok "new binary staged at $BINARY"

write_setdns "$CC_SETDNS"
[ -s "$CC_SETDNS" ] && ok "DNS resolver preload installed ($CC_SETDNS)" \
  || warn "could not write $CC_SETDNS; DNS ETIMEOUT workaround inactive."

# Smoke-test the new binary BEFORE removing the working install. If the latest
# Claude Code crashes on this device (a known upstream regression: Android 10's
# blocked statx raising SIGSYS from the seccomp filter, or the Bun 1.4
# epoll_pwait2 SIGSEGV, which is a NULL deref in Termux's glibc-runner shim, not
# a blocked syscall), abort and leave the current install untouched rather than
# migrating onto a binary that will not launch. The probe is --init-only, which
# boots the full runtime and exits 0 on a healthy binary; it passes --version yet
# crashes on full launch, so --version alone would not catch this.
info "smoke-testing the new binary"
ST_ERR="$VERSIONS_DIR/.smoke-stderr"
ST_HOME="$VERSIONS_DIR/.smoke-home"
ST_LIMIT="${CC_SMOKE_TIMEOUT:-45}"
rm -rf "$ST_HOME"; mkdir -p "$ST_HOME/.claude"
ST_STARTED="$(date +%s)"
if HOME="$ST_HOME" LD_PRELOAD='' timeout -s KILL "$ST_LIMIT" "$BINARY" --init-only </dev/null >/dev/null 2>"$ST_ERR"; then
  ST_RC=0
else
  ST_RC=$?
fi
ST_ELAPSED=$(( $(date +%s) - ST_STARTED ))
rm -rf "$ST_HOME"
if [ ! -s "$BINARY" ] \
   || grep -qE 'Bad system call|oh no: Bun has crashed|panic\(|bun\.report' "$ST_ERR" 2>/dev/null; then
  rm -f "$ST_ERR" "$BINARY"
  warn "Claude Code $LATEST crashes on this device. This is a known upstream"
  warn "regression, not a problem with your setup."
  warn "Your current install has NOT been changed. To get a working Claude Code:"
  warn "  - keep using your current install, or"
  warn "  - run  ./install-pinned.sh   to pin a known-good build, or"
  warn "  - run Claude Code inside proot-distro Ubuntu (see the README)."
  fail "migration aborted: the latest Claude Code does not run on this device"
elif [ "$ST_ELAPSED" -ge "$ST_LIMIT" ]; then
  rm -f "$ST_ERR"
  warn "Could not fully verify Claude Code $LATEST on this device: the launch"
  warn "probe timed out, which can happen on a slow or loaded device. Continuing"
  warn "the migration; the launcher re-checks it on first run and rolls back on"
  warn "its own if it does not start."
elif { [ "$ST_RC" -gt 128 ] && [ "$ST_RC" -le 159 ]; }; then
  rm -f "$ST_ERR" "$BINARY"
  warn "Claude Code $LATEST crashes on this device. This is a known upstream"
  warn "regression, not a problem with your setup."
  warn "Your current install has NOT been changed. To get a working Claude Code:"
  warn "  - keep using your current install, or"
  warn "  - run  ./install-pinned.sh   to pin a known-good build, or"
  warn "  - run Claude Code inside proot-distro Ubuntu (see the README)."
  fail "migration aborted: the latest Claude Code does not run on this device"
else
  rm -f "$ST_ERR"
  printf '%s\n' "$LATEST" > "$VERSIONS_DIR/.verified"
  ok "new binary launches cleanly on this device"
fi

# --- Remove the old install (only now that the new binary is verified) ---
# The official native install has no npm package and no $PREFIX/bin symlink, so
# for state=native this whole block is a no-op; the wrapper is written next.
if [ "$state" = pinned ]; then
  info "removing the old pinned v2.x install"
  if [ -d "$NPM_PKG" ]; then chmod -R u+w "$NPM_PKG" 2>/dev/null || true; fi
  if command -v npm >/dev/null 2>&1; then npm uninstall -g @anthropic-ai/claude-code >/dev/null 2>&1 || true; fi
  if [ -d "$NPM_PKG" ]; then rm -rf "$NPM_PKG" 2>/dev/null || true; fi
  if [ -L "$BINLINK" ]; then
    case "$(readlink "$BINLINK")" in
      *node_modules/@anthropic-ai/claude-code*) rm -f "$BINLINK" ;;
    esac
  fi
  ok "old install removed"
else
  ok "no npm package to remove (native install)"
fi

# --- Wrapper at $PREFIX/bin/claude  (KEEP BYTE-IDENTICAL TO install.sh) ---
# SYNC:BEGIN wrapper-heredoc (kept byte-identical to install.sh: checked by scripts/check-sync.sh)
cat > "$WRAPPER" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
VERSIONS_DIR="$VERSIONS_DIR"
GLIBC_LD="$GLIBC_LD"
PATCHELF="$PATCHELF"
STAMP="\$VERSIONS_DIR/.last-update-check"
BLOCKLIST="\$VERSIONS_DIR/.blocklist"
VERIFIED="\$VERSIONS_DIR/.verified"
RATE_LIMIT=86400

retry_update_soon() {
  retry_at=\$(( \$(date +%s) - RATE_LIMIT + 3600 ))
  touch -d "@\$retry_at" "\$STAMP" 2>/dev/null || rm -f "\$STAMP"
}

CC_SETDNS="$HOME/.local/share/claude/setdns.js"
CC_SETDNS_JS='try { require("dns").setServers(["8.8.8.8", "8.8.4.4"]); } catch (e) {}'
write_setdns() {
  [ -s "\$1" ] && return 0
  printf '%s\n' "\$CC_SETDNS_JS" > "\$1" 2>/dev/null
}

# Smoke test: returns 0 if the binary launches on this device, 1 if it
# DEFINITELY crashes here (a fatal signal or a known Bun/seccomp crash banner),
# and 2 if the result is inconclusive (the probe timed out, could not exec, or
# the file is empty). Why this exists: upstream has shipped binaries that pass
# "--version" but die on full launch, either from Android's seccomp filter
# (Android 10 statx or pidfd_open -> SIGSYS) or from a null deref in Termux's
# glibc-runner epoll_pwait2 shim under Bun 1.4 (-> SIGSEGV). We probe the full
# runtime with --init-only (it boots the HTTP thread and worker pool and exits
# 0 offline on a healthy binary) and refuse to promote or run anything that
# dies. Only a DEFINITE crash (return 1) is ever blocklisted; an inconclusive
# result (return 2, e.g. a probe that timed out on a slow or thermally
# throttled device) is never blocklisted, so a good build is not permanently
# rejected by a transient hiccup. If a future release drops --init-only the
# probe exits a benign non-zero with no signal and no crash banner, treated as
# healthy (return 0): not rejected, never a false fail.
smoke_test() {
  st_err="\$VERSIONS_DIR/.smoke-stderr"
  st_home="\$VERSIONS_DIR/.smoke-home"
  if [ ! -s "\$1" ]; then return 2; fi
  # Probe in an isolated HOME so we never load the user's hooks (--init-only
  # fires SessionStart/SessionEnd), never depend on login, and never write to
  # the real ~/.claude. The crash we detect is a syscall, independent of config.
  rm -rf "\$st_home"; mkdir -p "\$st_home/.claude"
  st_limit="\${CC_SMOKE_TIMEOUT:-45}"
  st_started=\$(date +%s)
  HOME="\$st_home" LD_PRELOAD= timeout -s KILL "\$st_limit" "\$1" --init-only </dev/null >/dev/null 2>"\$st_err"
  st_rc=\$?
  st_elapsed=\$(( \$(date +%s) - st_started ))
  rm -rf "\$st_home"
  # A known crash banner is authoritative even if it appeared near the timeout.
  if grep -qE 'Bad system call|oh no: Bun has crashed|panic\(|bun\.report' "\$st_err" 2>/dev/null; then
    rm -f "\$st_err"; return 1
  fi
  # timeout exit conventions vary. Elapsed time is the portable signal.
  if [ "\$st_elapsed" -ge "\$st_limit" ]; then rm -f "\$st_err"; return 2; fi
  if [ "\$st_rc" -gt 128 ] && [ "\$st_rc" -le 159 ]; then rm -f "\$st_err"; return 1; fi
  if [ "\$st_rc" -eq 126 ] || [ "\$st_rc" -eq 127 ]; then rm -f "\$st_err"; return 2; fi
  rm -f "\$st_err"
  return 0
}

force_update=0
args=()
for a in "\$@"; do
  if [ "\$a" = "--update-now" ]; then
    force_update=1
  else
    args+=("\$a")
  fi
done

should_check=0
if [ "\$force_update" = 1 ]; then
  should_check=1
elif [ ! -f "\$STAMP" ]; then
  should_check=1
else
  now=\$(date +%s)
  last=\$(stat -c%Y "\$STAMP" 2>/dev/null || echo 0)
  [ \$((now - last)) -ge \$RATE_LIMIT ] && should_check=1
fi

if [ "\$should_check" = 1 ]; then
  # One-updater lock: only one claude process downloads at a time. A second
  # launch during the (up to 5 min) download skips the update and runs the
  # cached binary instead of racing on a shared staging file. A crashed
  # updater's lock is stolen after 15 min so updates can never wedge forever.
  LOCK="\$VERSIONS_DIR/.update.lock"
  if [ -d "\$LOCK" ]; then
    lock_age=\$(( \$(date +%s) - \$(stat -c%Y "\$LOCK" 2>/dev/null || echo 0) ))
    [ "\$lock_age" -ge 900 ] && rmdir "\$LOCK" 2>/dev/null
  fi
  if mkdir "\$LOCK" 2>/dev/null; then
    # A SIGKILL during download bypasses normal cleanup. Sweep stale staging
    # files on every serialized update check; a live download is only minutes old.
    cleanup_now=\$(date +%s 2>/dev/null || echo "")
    case "\$cleanup_now" in ""|*[!0-9]*) cleanup_now=0 ;; esac
    for stale_tmp in "\$VERSIONS_DIR"/*.tmp; do
      [ -f "\$stale_tmp" ] && [ ! -L "\$stale_tmp" ] || continue
      stale_mtime=\$(stat -c%Y "\$stale_tmp" 2>/dev/null || echo "")
      case "\$stale_mtime" in ""|*[!0-9]*) continue ;; esac
      [ \$(( cleanup_now - stale_mtime )) -gt 86400 ] && rm -f "\$stale_tmp" 2>/dev/null
    done
    touch "\$STAMP"
    latest=\$(curl -fsSL --max-time 5 https://registry.npmjs.org/@anthropic-ai/claude-code/latest 2>/dev/null | jq -r .version 2>/dev/null || echo "")
    if [ -n "\$latest" ] && printf '%s' "\$latest" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\$'; then
      new_bin="\$VERSIONS_DIR/\$latest"
      # Per-process staging path (never a shared name) so two updaters cannot
      # clobber each other's in-flight download.
      tmp="\$new_bin.\$\$.tmp"
      if [ ! -f "\$new_bin" ] && ! grep -qxF "\$latest" "\$BLOCKLIST" 2>/dev/null; then
        dl="https://downloads.claude.ai/claude-code-releases/\$latest"
        if curl -fsSL --max-time 300 "\$dl/linux-arm64/claude" -o "\$tmp" 2>/dev/null && [ -s "\$tmp" ]; then
          exp=\$(curl -fsSL --max-time 5 "\$dl/manifest.json" 2>/dev/null | jq -er '.platforms["linux-arm64"].checksum' 2>/dev/null || echo "")
          act=\$(sha256sum "\$tmp" 2>/dev/null | cut -d' ' -f1)
          if [ -z "\$exp" ]; then
            rm -f "\$tmp"
            retry_update_soon
            echo "[claude] update: could not read release manifest, using cached" >&2
          elif [ "\$exp" != "\$act" ]; then
            rm -f "\$tmp"
            retry_update_soon
            echo "[claude] update: checksum mismatch on \$latest, using cached" >&2
          else
            chmod +x "\$tmp"
            if ! LD_PRELOAD= "\$PATCHELF" --set-interpreter "\$GLIBC_LD" "\$tmp" 2>/dev/null; then
              rm -f "\$tmp"
              echo "[claude] update: patchelf failed on \$latest, using cached" >&2
            else
              smoke_test "\$tmp"; sc=\$?
              if [ "\$sc" -eq 0 ]; then
                mv "\$tmp" "\$new_bin"
                printf '%s\n' "\$latest" > "\$VERIFIED"
                # Retain N-1 (latest + previous) for rollback. Only version-named
                # binaries are removed, never a staging .tmp or the lock dir.
                prev=\$(ls -1 "\$VERSIONS_DIR" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\$' | sort -V | tail -2 | head -1)
                for old in "\$VERSIONS_DIR"/*; do
                  base=\$(basename "\$old")
                  printf '%s' "\$base" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\$' || continue
                  [ -f "\$old" ] && [ "\$base" != "\$latest" ] && [ "\$base" != "\$prev" ] && rm -f "\$old"
                done
              elif [ "\$sc" -eq 1 ]; then
                rm -f "\$tmp"
                printf '%s\n' "\$latest" >> "\$BLOCKLIST"
                echo "[claude] update: \$latest crashes on launch (failed smoke test), keeping cached" >&2
              else
                rm -f "\$tmp"
                echo "[claude] update: could not verify \$latest on this device, keeping cached" >&2
              fi
            fi
          fi
        else
          rm -f "\$tmp" 2>/dev/null
          retry_update_soon
          echo "[claude] update: download incomplete, using cached" >&2
        fi
      fi
    else
      retry_update_soon
      echo "[claude] update: could not query npm registry, using cached" >&2
    fi
    rmdir "\$LOCK" 2>/dev/null
  fi
fi

# Pick the highest installed version that actually launches on this device.
# Self-healing rollback: skip blocklisted versions; the already-verified-good
# version runs with no re-test (zero startup cost); any other candidate is
# re-patched and smoke-tested, and if it crashes it is blocklisted and we fall
# back to the next-highest. This rescues a device that auto-updated to a binary
# that crashes here (e.g. a bad release that landed before this wrapper shipped)
# with no user action.
verified=\$(cat "\$VERIFIED" 2>/dev/null || echo "")
bin=""
fallback=""
for cand in \$(ls -1 "\$VERSIONS_DIR" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\$' | sort -Vr); do
  grep -qxF "\$cand" "\$BLOCKLIST" 2>/dev/null && continue
  cpath="\$VERSIONS_DIR/\$cand"
  [ -f "\$cpath" ] || continue
  if [ "\$cand" = "\$verified" ]; then bin="\$cpath"; break; fi
  interp=\$(LD_PRELOAD= "\$PATCHELF" --print-interpreter "\$cpath" 2>/dev/null || echo unknown)
  [ "\$interp" = "\$GLIBC_LD" ] || LD_PRELOAD= "\$PATCHELF" --set-interpreter "\$GLIBC_LD" "\$cpath" 2>/dev/null
  smoke_test "\$cpath"; sc=\$?
  if [ "\$sc" -eq 0 ]; then
    printf '%s\n' "\$cand" > "\$VERIFIED"
    bin="\$cpath"
    break
  elif [ "\$sc" -eq 1 ]; then
    echo "[claude] \$cand crashes on this device; rolling back to the previous version" >&2
    printf '%s\n' "\$cand" >> "\$BLOCKLIST"
  else
    # Inconclusive (e.g. the probe timed out on a slow device): do not blocklist,
    # but remember the highest such build as a last resort so we still launch.
    [ -z "\$fallback" ] && fallback="\$cpath"
    echo "[claude] could not verify \$cand on this device; trying an older version first" >&2
  fi
done
# Nothing probed clean, but a build merely failed to prove itself (never
# crashed): run the highest such build rather than refuse. An inconclusive
# probe is not a crash.
[ -z "\$bin" ] && [ -n "\$fallback" ] && bin="\$fallback"
if [ -z "\$bin" ]; then
  echo "[claude] no working claude binary found in \$VERSIONS_DIR. Re-run install.sh." >&2
  exit 1
fi

write_setdns "\$CC_SETDNS"
if [ -s "\$CC_SETDNS" ]; then
  # Bun resolves relative preloads from its physical CWD; default realpath resolves
  # symlinks to match that assumption and avoids its node_modules walk to / (cosmetic EACCES).
  # The absolute fallback keeps DNS working if relative-path resolution fails.
  cc_preload=\$(realpath --relative-to="\$PWD" "\$CC_SETDNS" 2>/dev/null) || cc_preload=""
  case "\$cc_preload" in
    ./*|../*) ;;
    "") cc_preload="\$CC_SETDNS" ;;
    /*) cc_preload="\$CC_SETDNS" ;;
    *) cc_preload="./\$cc_preload" ;;
  esac
  export BUN_OPTIONS="--preload \$cc_preload\${BUN_OPTIONS:+ \$BUN_OPTIONS}"
fi
unset LD_PRELOAD
exec "\$bin" "\${args[@]}"
EOF
# SYNC:END wrapper-heredoc
chmod +x "$WRAPPER"
ok "wrapper installed at $WRAPPER"

# --- Native-install launcher discovery ---
# Claude Code sees the binary under ~/.local/share/claude/versions, treats it as
# a native install, and expects a launcher at ~/.local/bin/claude with
# ~/.local/bin on PATH. Without them it prints "Native installation ... not in
# your PATH" notices at startup. Set both up the way claude's own message
# prescribes. The launcher points at this wrapper so every invocation still
# routes through it; ~/.local/bin is appended to PATH so $PREFIX/bin stays first.
mkdir -p "$HOME/.local/bin"
ln -sfn "$WRAPPER" "$HOME/.local/bin/claude"
if ! grep -Fq 'native-install launcher discovery' "$HOME/.bashrc" 2>/dev/null; then
  printf '\n# claude-code-android: native-install launcher discovery\nexport PATH="$PATH:$HOME/.local/bin"\n' >> "$HOME/.bashrc"
  ok "added ~/.local/bin to PATH in ~/.bashrc"
else
  ok "PATH already includes ~/.local/bin in ~/.bashrc"
fi

# --- Merge settings.json (symlink-safe; preserve existing keys) ---
# autoUpdates:false hands updates to the wrapper. No env.LD_PRELOAD: a bionic
# preload there leaks into the Bash tool's subprocesses and breaks claude's
# bundled grep/rg/ugrep. Any stale LD_PRELOAD from an earlier version is removed.
# Known trade-off: without the preload, claude's subprocesses also lose
# termux-exec, so a directly-run "#!/usr/bin/env ..." script cannot find its
# interpreter (Android has no /usr/bin/env). Grep correctness wins.
SF="$HOME/.claude/settings.json"
if [ -e "$SF" ]; then
  TMP="$(mktemp "${TMPDIR:-$PREFIX/tmp}/cc-settings.XXXXXX")"
  if jq 'del(.env.LD_PRELOAD) | .autoUpdates=false | if (.env // {}) == {} then del(.env) else . end' "$SF" > "$TMP" 2>/dev/null; then
    # Write THROUGH the file (cat, not mv) so a symlink is followed, not replaced.
    cat "$TMP" > "$SF"
    rm -f "$TMP"
    if [ -L "$SF" ]; then
      warn "settings.json is a symlink -> $(readlink -f "$SF"). Updated the target in place; if it is version-controlled, review and commit the change."
    fi
    ok "settings.json merged (your existing keys preserved; stale LD_PRELOAD removed)"
  else
    rm -f "$TMP"
    warn "settings.json is not valid JSON; leaving it untouched to avoid corrupting it."
    warn "Set  \"autoUpdates\": false  by hand and remove any env.LD_PRELOAD."
  fi
else
  cat > "$SF" <<'SET'
{
  "autoUpdates": false
}
SET
  ok "settings.json written"
fi

# --- Recommended packages ---
if [ "$RECOMMENDED" = 1 ]; then
  info "installing recommended packages (this is the longest step)"
  apt-get install $APT_OPTS git gh wget jq python openssh tree proot \
    termux-api proot-distro make clang file xxd htop bat fzf >/dev/null \
    || fail "recommended package install failed"
  ok "recommended packages installed"
fi

# --- Verify ---
hash -r 2>/dev/null || true
VER="$(claude --version 2>&1)" || fail "claude --version failed: $VER"
RES="$(command -v claude || true)"
if [ "$RES" != "$WRAPPER" ]; then warn "claude resolves to $RES (expected $WRAPPER)"; fi
# Count preserved sessions. Guard the directory: a claude that was installed
# but never launched has no projects/ yet, and under 'set -e' a bare ls on a
# missing path would abort the run at the very end (after the real work is done).
if [ -d "$HOME/.claude/projects" ]; then
  SESS="$(ls -1 "$HOME/.claude/projects" 2>/dev/null | wc -l | tr -d ' ')"
else
  SESS=0
fi
ok "claude --version: $VER"

# --- bashrc stale-line detection (suggest only; never auto-edit) ---
STALE="$(grep -nE 'DISABLE_AUTOUPDATER=1|claude-android|CLAUDE_CODE_USE_NATIVE_FILE_SEARCH=1' "$HOME/.bashrc" 2>/dev/null || true)"

trap - ERR
cat <<DONE

Migration complete.

  Now on:    $VER
  Wrapper:   $WRAPPER
  Binary:    $BINARY
  Sessions:  $SESS preserved
  Backup:    $BACKUP_DIR
             (restore anytime with: bash $BACKUP_DIR/restore.sh)

Your login and chats are intact. The wrapper auto-checks for a new claude
release once per day on launch. Force a check with: claude --update-now

Open a new Termux session (so the updated PATH is active and startup is
warning-free), then type:

  claude

DONE

if [ -n "$STALE" ]; then
  cat <<NOTE
Optional cleanup: these old v2.x lines in ~/.bashrc are now harmless under
the current architecture and can be removed by hand if you like (leave anything else alone):
$STALE

NOTE
fi
