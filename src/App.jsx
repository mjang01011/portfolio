import { Routes, Route } from 'react-router-dom';
import Home from './Components/Home/Home';
// import IFrame from './utils/IFrame';

const App = () => {
  return (
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
  );
};

export default App;
