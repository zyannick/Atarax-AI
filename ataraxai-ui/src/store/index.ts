import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { ProjectSlice, createProjectSlice } from './projectSlice';
import { SessionSlice, createSessionSlice } from './sessionSlice';
import { MessageSlice, createMessageSlice } from './messageSlice';
import { UISlice, createUISlice } from './uiSlice';

export type AppStore = ProjectSlice & SessionSlice & MessageSlice & UISlice;

export const useAppStore = create<AppStore>()(
  devtools(
    (...args) => ({
      ...createProjectSlice(...args),
      ...createSessionSlice(...args),
      ...createMessageSlice(...args),
      ...createUISlice(...args),
    }),
    {
      name: 'atarax-ai-store',
    }
  )
);