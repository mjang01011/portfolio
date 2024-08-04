import "./BlogNavBar.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import { Link } from "react-router-dom";

const BlogNavBar = () => {

  return (
    <div className="blog-navbar">
      <ul className="blog-nav-menu">
        <Link className="blog-nav-link" to="/">
          <li>Home</li>
        </Link>
        <Link className="blog-nav-link" to="/blogs">
          <li>Blogs</li>
        </Link>
      </ul>
      <div className="blog-nav-logo">
        <Link
          className="blog-nav-link"
          to="https://www.linkedin.com/in/michaeljkjang/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src={logo_linkedin} alt="LinkedIn" />
        </Link>
        <Link
          className="blog-nav-link"
          to="https://github.com/mjang01011"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src={logo_github} alt="GitHub" />
        </Link>
      </div>
    </div>
  );
};

export default BlogNavBar;
