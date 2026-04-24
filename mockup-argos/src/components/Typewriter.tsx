import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

interface Props {
  text: string;
  speed?: number;
  className?: string;
  startDelay?: number;
}

export function Typewriter({ text, speed = 28, className, startDelay = 0 }: Props) {
  const [shown, setShown] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  useEffect(() => {
    if (!inView) return;
    let i = 0;
    const start = setTimeout(() => {
      const id = setInterval(() => {
        i++;
        setShown(i);
        if (i >= text.length) clearInterval(id);
      }, speed);
    }, startDelay);
    return () => clearTimeout(start);
  }, [inView, text, speed, startDelay]);

  return (
    <span ref={ref} className={className}>
      {text.slice(0, shown)}
      {shown < text.length && <span className="typer-caret" />}
    </span>
  );
}
