import "./AllBlogs.css";
import theme_pattern from "../../assets/brush_skyblue.png";
import notebook_data from "../../assets/notebook_data";
import markdown_data from "../../assets/markdown_data";
import paper_data from "../../assets/paper_data";
import { Link } from "react-router-dom";
import { useEffect } from "react";

const AllBlogs = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div id="all-blogs" className="blog">
      <div className="blog-title">
        <h1>Blogs</h1>
        <img src={theme_pattern} alt="" />
      </div>
      <h2 className="section-title">Paper Summary</h2>
      <div className="blog-container">
        {paper_data.map((ppt, index) => {
          return (
            <div key={index} className="blog-wrapper">
              <a href={ppt.link} target="_blank" rel="noopener noreferrer" className="blog-link">
                <h2>{ppt.name}</h2>
                <ul>
                  {ppt.contents.map((content, content_index) => (
                    <li key={content_index}>{content}</li>
                  ))}
                </ul>
              </a>
            </div>
          );
        })}
      </div>
      <h2 className="section-title">Jupyter Notebook Implementation</h2>
      <div className="blog-container">
        {notebook_data.map((blog, index) => {
          return (
            <div key={index} className="blog-wrapper">
              <Link className="blog-link" to={"/blogs/notebooks/" + blog.link}>
                <h2>{blog.name}</h2>
                <ul>
                  {blog.contents.map((content, content_index) => (
                    <li key={content_index}>{content}</li>
                  ))}
                </ul>
              </Link>
            </div>
          );
        })}
      </div>
      <h2 className="section-title">Markdown Blogs</h2>
      <div className="blog-container">
        {markdown_data.map((markdown, index) => {
          return (
            <div key={index} className="blog-wrapper">
              <Link className="blog-link" to={"/blogs/markdowns/" + markdown.link}>
                <h2>{markdown.name}</h2>
                <ul>
                  {markdown.contents.map((content, content_index) => (
                    <li key={content_index}>{content}</li>
                  ))}
                </ul>
              </Link>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AllBlogs;
