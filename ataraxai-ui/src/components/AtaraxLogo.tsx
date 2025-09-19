import React from 'react';

interface AtaraxLogoProps {
  size?: number;
  className?: string;
}

export function AtaraxLogo({ size = 24, className = '' }: AtaraxLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Minimalist geometric owl design */}
      <path
        d="M12 2C8 2 5 5 5 9v6c0 2 1 3 2 4l1 1v1c0 1 1 2 2 2h4c1 0 2-1 2-2v-1l1-1c1-1 2-2 2-4V9c0-4-3-7-7-7z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Eyes */}
      <circle cx="9" cy="11" r="1.5" fill="currentColor" />
      <circle cx="15" cy="11" r="1.5" fill="currentColor" />
      {/* Beak */}
      <path
        d="M12 13v2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      {/* Ear tufts */}
      <path
        d="M8.5 4.5l-1-2M15.5 4.5l1-2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}