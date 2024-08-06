import "./Blogs.css";
import theme_pattern from "../../assets/brush_skyblue.png";
import notebook_data from "../../assets/notebook_data";
import arrow_icon from "../../assets/arrow_icon.svg";
import { Link } from "react-router-dom";

const Blog = () => {
  return (
    <div id="blog" className="blog">
      <div className="blog-title">
        <h1>Blogs</h1>
        <img src={theme_pattern} alt="Theme pattern" />
      </div>
      <div className="blog-container" id="blog-container">
        {notebook_data.map((blog, index) => (
          <div key={index} className="blog-wrapper">
            <Link className="blog-link" to={"/blogs/" + blog.link}>
              <h2>{blog.name}</h2>
              {blog.contents && blog.contents.length > 0 && (
                <ul>
                  {blog.contents.map((content, content_index) => (
                    <li key={content_index}>{content}</li>
                  ))}
                </ul>
              )}
            </Link>
          </div>
        ))}
      </div>
      <Link className="blog-link" to="/blogs">
        <div className="blog-showmore">
          <p>View my blogs</p>
          <img src={arrow_icon} alt="Arrow icon" />
        </div>
      </Link>
    </div>
  );
};

export default Blog;
