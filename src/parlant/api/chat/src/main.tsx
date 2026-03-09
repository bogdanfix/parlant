import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import {Toaster} from './components/ui/sonner.tsx';
import { getBasePath } from './utils/api.ts';

// Dynamically create favicon using BASE_URL
const favicon = document.createElement('link');
favicon.rel = 'icon';
favicon.type = 'image/svg+xml';
favicon.href = `${getBasePath()}/chat/logo-color.svg`;
document.head.appendChild(favicon);

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <App />
        <Toaster position='bottom-center' toastOptions={{className: 'rounded-full w-fit px-[34px] !bg-[#006E54] text-white'}} className='mb-[80px] transition-none animate-none rounded-full' />
    </StrictMode>
);