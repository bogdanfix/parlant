import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import {Toaster} from './components/ui/sonner.tsx';
import { getBasePath } from './utils/api.ts';

// Dynamically set favicon using BASE_URL
const favicon = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
if (favicon) {
  favicon.href = `${getBasePath()}/chat/logo-color.svg`;
}

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <App />
        <Toaster position='bottom-center' toastOptions={{className: 'rounded-full w-fit px-[34px] !bg-[#006E54] text-white'}} className='mb-[80px] transition-none animate-none rounded-full' />
    </StrictMode>
);