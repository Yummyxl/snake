import { createContext, useCallback, useMemo, useRef, useState } from "react";

export const UiContext = createContext(null);

function nextId(ref) {
  ref.current += 1;
  return ref.current;
}

function ToastViewport({ items, onDismiss }) {
  return (
    <div className="toastViewport" aria-live="polite" aria-relevant="additions removals">
      {items.map((t) => (
        <div key={t.id} className={`toast toast--${t.kind}`}>
          <div className="toast__msg">{t.message}</div>
          <button className="toast__x" type="button" onClick={() => onDismiss(t.id)} aria-label="关闭">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

function ConfirmDialog({ open, title, message, confirmText, cancelText, onConfirm, onCancel }) {
  if (!open) return null;
  return (
    <div className="modalOverlay" role="dialog" aria-modal="true" aria-label={title || "确认"}>
      <div className="modal">
        <div className="modal__title">{title || "确认"}</div>
        <div className="modal__body">{message}</div>
        <div className="modal__actions">
          <button className="btn" type="button" onClick={onCancel}>{cancelText || "取消"}</button>
          <button className="btn btn--danger" type="button" onClick={onConfirm}>{confirmText || "确定"}</button>
        </div>
      </div>
    </div>
  );
}

export default function UiProvider({ children }) {
  const idRef = useRef(0);
  const [toasts, setToasts] = useState([]);
  const [confirmState, setConfirmState] = useState(null);

  const dismissToast = useCallback((id) => {
    setToasts((xs) => xs.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback((message, kind = "info", timeoutMs = 800) => {
    const id = nextId(idRef);
    const t = { id, kind, message: String(message) };
    setToasts((xs) => [...xs, t].slice(-4));
    window.setTimeout(() => dismissToast(id), timeoutMs);
  }, [dismissToast]);

  const confirm = useCallback(({ title, message, confirmText, cancelText }) => {
    return new Promise((resolve) => {
      setConfirmState({ title, message, confirmText, cancelText, resolve });
    });
  }, []);

  const onConfirm = useCallback(() => {
    setConfirmState((s) => { s?.resolve(true); return null; });
  }, []);

  const onCancel = useCallback(() => {
    setConfirmState((s) => { s?.resolve(false); return null; });
  }, []);

  const value = useMemo(() => ({ toast, confirm }), [toast, confirm]);

  return (
    <UiContext.Provider value={value}>
      {children}
      <ToastViewport items={toasts} onDismiss={dismissToast} />
      <ConfirmDialog
        open={Boolean(confirmState)}
        title={confirmState?.title}
        message={confirmState?.message}
        confirmText={confirmState?.confirmText}
        cancelText={confirmState?.cancelText}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    </UiContext.Provider>
  );
}
