import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import matter from "gray-matter";

const AllBlogs = () => {
  const [blogs, setBlogs] = useState([]);

  useEffect(() => {
    const importMarkdownFiles = () => {
      const markdownFiles = require.context("../../markdowns", false, /\.md$/);
      const fileNames = markdownFiles.keys();

      const blogList = fileNames.map((fileName) => {
        const markdownContent = markdownFiles(fileName).default;
        const { data } = matter(markdownContent);
        const id = fileName.replace("./", "").replace(".md", "");
        return { id, title: data.title };
      });

      setBlogs(blogList);
    };

    importMarkdownFiles();
  }, []);

  return (
    <div>
      <h1>All Blogs</h1>
      <ul>
        {blogs.map((blog) => (
          <li key={blog.id}>
            <Link to={`/blog/${blog.id}`}>{blog.title}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AllBlogs;
