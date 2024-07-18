import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import NavBar from "./Components/NavBar/NavBar";
import Hero from "./Components/Hero/Hero";
import About from "./Components/About/About";
import Blogs from "./Components/Blogs/Blogs";
import MyWork from "./Components/MyWork/MyWork";
import AllBlogs from "./Components/AllBlogs/AllBlogs";
import BlogPost from "./Components/BlogPost/BlogPost";

const App = () => {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route
          path="/"
          element={
            <>
              <Hero />
              <About />
              {/* <Blogs /> */}
              <MyWork />
            </>
          }
        />
        {/* <Route path="/all-blogs" element={<AllBlogs />} /> */}
        {/* <Route path="/blog/:id" element={<BlogPost />} /> */}
      </Routes>
    </Router>
  );
};

export default App;
