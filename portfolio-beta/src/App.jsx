import { Routes, Route, useLocation } from "react-router-dom";
import NavBar from "./Components/NavBar/NavBar";
import Home from "./Components/Home/Home";
import AllBlogs from "./Components/AllBlogs/AllBlogs";
import BlogNavBar from "./Components/NavBar/BlogNavBar";
import Footer from "./Components/Footer/Footer";
import NotebookViewer from "./utils/NotebookViewer";
import MarkdownRenderer from "./utils/MarkdownRenderer";

const App = () => {
  const location = useLocation();
  const isBlogRoute = location.pathname.startsWith("/blogs");

  return (
    <>
      {isBlogRoute ? <BlogNavBar isNotebook={location.pathname.includes("/notebooks/")} /> : <NavBar />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blogs" element={<AllBlogs />} />
        <Route path="/blogs/notebooks/:filename" element={<NotebookViewer />} />
        <Route path="/blogs/markdowns/:filename" element={<MarkdownRenderer />} />
      </Routes>
      <Footer />
    </>
  );
};

export default App;
