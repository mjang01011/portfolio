import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import App from './App';
import './index.css';

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);

root.render(
  <HashRouter basename="/portfolio">
    <App />
  </HashRouter>
);
