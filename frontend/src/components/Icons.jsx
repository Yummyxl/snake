function Svg({ children, title, ...props }) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden={title ? undefined : true} role="img" {...props}>
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

export function IconPlay(props) {
  return (
    <Svg {...props}>
      <path d="M7.4 5.8c0-.66.73-1.06 1.28-.71l6.3 4.2a.83.83 0 0 1 0 1.42l-6.3 4.2c-.55.36-1.28-.04-1.28-.7V5.8z" fill="currentColor" />
    </Svg>
  );
}

export function IconPause(props) {
  return (
    <Svg {...props}>
      <path d="M6.6 5.6c0-.44.36-.8.8-.8h1c.44 0 .8.36.8.8v8.8c0 .44-.36.8-.8.8h-1c-.44 0-.8-.36-.8-.8V5.6zm5 0c0-.44.36-.8.8-.8h1c.44 0 .8.36.8.8v8.8c0 .44-.36.8-.8.8h-1c-.44 0-.8-.36-.8-.8V5.6z" fill="currentColor" />
    </Svg>
  );
}

export function IconPrev(props) {
  return (
    <Svg {...props}>
      <path d="M12.4 4.8 7.4 10l5 5.2" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconFirst(props) {
  return (
    <Svg {...props}>
      <path d="M6.2 5.0c0-.44.36-.8.8-.8s.8.36.8.8v10c0 .44-.36.8-.8.8s-.8-.36-.8-.8V5.0z" fill="currentColor" />
      <path d="M14.0 4.8 9.0 10l5 5.2" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconNext(props) {
  return (
    <Svg {...props}>
      <path d="M7.6 4.8 12.6 10l-5 5.2" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconStop(props) {
  return (
    <Svg {...props}>
      <path d="M6.8 6.8c0-.55.45-1 1-1h4.4c.55 0 1 .45 1 1v4.4c0 .55-.45 1-1 1H7.8c-.55 0-1-.45-1-1V6.8z" fill="currentColor" />
    </Svg>
  );
}

export function IconList(props) {
  return (
    <Svg {...props}>
      <path d="M5 6.2h10v1.6H5V6.2zm0 3h10v1.6H5V9.2zm0 3h10v1.6H5v-1.6z" fill="currentColor" />
    </Svg>
  );
}

export function IconRefresh(props) {
  return (
    <Svg {...props}>
      <path d="M15.6 10a5.6 5.6 0 1 1-1.06-3.3l.76-.76v3.06h-3.06l.98-.98A4 4 0 1 0 14.4 10h1.2z" fill="currentColor" />
    </Svg>
  );
}

export function IconX(props) {
  return (
    <Svg {...props}>
      <path d="M6.2 6.2a.8.8 0 0 1 1.13 0L10 8.87l2.67-2.67a.8.8 0 1 1 1.13 1.13L11.13 10l2.67 2.67a.8.8 0 1 1-1.13 1.13L10 11.13 7.33 13.8a.8.8 0 1 1-1.13-1.13L8.87 10 6.2 7.33a.8.8 0 0 1 0-1.13z" fill="currentColor" />
    </Svg>
  );
}
