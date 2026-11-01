import { useState } from "react";

import { Note, Panel } from "./ui";

export function PrivacyPanel({ onWiped }: { onWiped: () => void }) {
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState("");
  const [pin, setPin] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const wipe = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const response = await fetch("/api/data/wipe", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(pin ? { "X-Clinician-Pin": pin } : {}) },
        body: JSON.stringify({ confirm }),
      });
      const data = (await response.json().catch(() => ({}))) as { detail?: string };
      if (!response.ok) throw new Error(data.detail ?? "Wipe failed.");
      try {
        Object.keys(localStorage)
          .filter((key) => key.startsWith("clinura."))
          .forEach((key) => localStorage.removeItem(key));
      } catch {
        // storage unavailable; server data is already gone
      }
      setMessage("All data wiped and the encryption key replaced.");
      setOpen(false);
      setConfirm("");
      setPin("");
      onWiped();
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Wipe failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Privacy and data" meta="Encrypted on this computer">
      <p className="muted">
        Readings, events, exercise plans and sessions are stored only on this computer, encrypted with AES-256. The
        key is kept in your system keychain. Video is never saved or sent anywhere.
      </p>
      {!open ? (
        <div className="panel-footer">
          <button type="button" className="btn btn--danger" onClick={() => setOpen(true)}>
            Wipe all data
          </button>
        </div>
      ) : (
        <div className="stack wipe-box">
          <Note>
            This permanently deletes all readings, events, notes, exercise plans, sessions and your local profile, and
            replaces the encryption key so old copies cannot be read. It cannot be undone.
          </Note>
          <div className="form-row">
            <label className="field">
              <span>Type WIPE to confirm</span>
              <input value={confirm} onChange={(e) => setConfirm(e.target.value)} autoFocus />
            </label>
            <label className="field">
              <span>Clinician PIN (if set)</span>
              <input type="password" inputMode="numeric" value={pin} onChange={(e) => setPin(e.target.value)} />
            </label>
          </div>
          <div className="dialog-actions">
            <button type="button" className="btn btn--quiet" onClick={() => setOpen(false)}>
              Cancel
            </button>
            <button
              type="button"
              className="btn btn--danger"
              disabled={confirm !== "WIPE" || busy}
              onClick={() => void wipe()}
            >
              {busy ? "Wiping…" : "Wipe everything"}
            </button>
          </div>
        </div>
      )}
      {message ? <Note>{message}</Note> : null}
    </Panel>
  );
}
