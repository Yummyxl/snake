export default function Banner({ kind, children, action }) {
  return (
    <div className={`banner banner--${kind}`}>
      <div className="banner__content">{children}</div>
      {action}
    </div>
  );
}

