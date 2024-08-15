import { Routes, Route } from "react-router-dom";
import NavBar from "./Components/NavBar/NavBar";
import Home from "./Components/Home/Home";
import AllBlogs from "./Components/AllBlogs/AllBlogs";
import BlogNavBar from "./Components/NavBar/BlogNavBar";
import Footer from "./Components/Footer/Footer";
import NotebookViewer from "./utils/NotebookViewer";
import MarkdownRenderer from "./utils/MarkdownRenderer";

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
              <BlogNavBar isNotebook={false} /> <AllBlogs />
            </>
          }
        />
        <Route
          path="/blogs/notebooks/:filename"
          element={
            <>
              <BlogNavBar isNotebook={true} /> <NotebookViewer />
            </>
          }
        />
        <Route
          path="/blogs/markdowns/:filename"
          element={
            <>
              <BlogNavBar isNotebook={true} /> <MarkdownRenderer />
            </>
          }
        />
      </Routes>
      <Footer />
    </>
  );
};

export default App;
