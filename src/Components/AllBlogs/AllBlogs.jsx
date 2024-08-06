import "./AllBlogs.css";
import theme_pattern from "../../assets/brush_skyblue.png";
import notebook_data from "../../assets/notebook_data";
import { Link } from "react-router-dom";

const AllBlogs = () => {
  return (
    <div id="all-blogs" className="blog">
      <div className="blog-title">
        <h1>Blogs</h1>
        <img src={theme_pattern} alt="" />
      </div>
      <h2 className="section-title">Jupyter Notebook Implementation</h2>
      <div className="blog-container" id="blog-container">
        {notebook_data.map((blog, index) => {
          return (
            <div key={index} className="blog-wrapper">
              <Link className="blog-link" to={"/blogs/" + blog.link}>
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
    </div>
  );
};

export default AllBlogs;
