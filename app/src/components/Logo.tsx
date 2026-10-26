export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="8" className="logo-tile" />
      <path
        d="M6 17.5h5l2.2-6 3.6 10 2.4-6.5 1.4 2.5H26"
        fill="none"
        className="logo-trace"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
