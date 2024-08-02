import "./Blogs.css";
import theme_pattern from "../../assets/brush_skyblue.png";
import blog_data from "../../assets/blog_data";
import arrow_icon from "../../assets/arrow_icon.svg";
import { Link } from "react-router-dom";

const blog = () => {
  return (
    <div id="blog" className="blog">
      <div className="blog-title">
        <h1>Blogs</h1>
        <img src={theme_pattern} alt="" />
      </div>
      <div className="blog-container" id="blog-container">
        {blog_data.map((blog, index) => {
          return (
            <div key={index} className="blog-wrapper">
              <Link className="blog-link" to={"/blog/" + blog.link}>
                <h2>{blog.name}</h2>
              </Link>
            </div>
          );
        })}
      </div>
      <Link className="blog-link" to="/blogs">
        <div className="blog-showmore">
          <p>View my blogs</p>
          <img src={arrow_icon} alt="" />
        </div>
      </Link>
    </div>
  );
};

export default blog;
