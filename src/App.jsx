import { Routes, Route } from "react-router-dom";
import NavBar from "./Components/NavBar/NavBar";
import Home from "./Components/Home/Home";
import Blogs from "./Components/Blogs/Blogs";
import BlogNavBar from "./Components/NavBar/BlogNavBar";
import Footer from "./Components/Footer/Footer";
// import IFrame from './utils/IFrame';

const App = () => {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route
          path="/blogs"
          element={
            <>
              <BlogNavBar /> <Blogs />
            </>
          }
        />
      </Routes>
      <Footer />
    </>
  );
};

export default App;
