import React, { useState, useEffect } from 'react';

// Mock AtaraxLogo component
export const AtaraxLogo = ({ className }: { className?: string }) => (
  <div className={`flex items-center justify-center ${className}`}>
    <div className="w-full h-full bg-gradient-to-br from-amber-800 to-amber-600 rounded-lg" />
  </div>
);

export default function LoadingScreen() {
  const [dots, setDots] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [status, setStatus] = useState('Initializing backend');

  useEffect(() => {
    // Animate dots
    const dotsInterval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);

    return () => clearInterval(dotsInterval);
  }, []);

  useEffect(() => {
    // Track elapsed time and update status messages
    const timeInterval = setInterval(() => {
      setElapsed(prev => {
        const next = prev + 1;
        
        if (next === 5) setStatus('Starting Python backend');
        else if (next === 10) setStatus('Loading AI models');
        else if (next === 20) setStatus('Establishing connection');
        else if (next === 30) setStatus('This is taking longer than expected');
        
        return next;
      });
    }, 1000);

    return () => clearInterval(timeInterval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-amber-50 to-stone-100">
      <AtaraxLogo className="h-20 w-20 animate-pulse" />
      
      <div className="mt-6 text-center">
        <p className="text-lg text-amber-900 font-medium">
          {status}{dots}
        </p>
        
        <div className="mt-4 flex items-center gap-2">
          <div className="w-48 h-1 bg-amber-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-amber-600 transition-all duration-1000 ease-out"
              style={{ 
                width: `${Math.min((elapsed / 30) * 100, 100)}%` 
              }}
            />
          </div>
          <span className="text-sm text-amber-700 w-12 text-right">
            {elapsed}s
          </span>
        </div>
        
        {elapsed > 30 && (
          <p className="mt-4 text-xs text-amber-700 max-w-md">
            The backend is taking longer than usual to start. This can happen on first launch
            or if the system is under heavy load. Please wait...
          </p>
        )}
      </div>
    </div>
  );
}