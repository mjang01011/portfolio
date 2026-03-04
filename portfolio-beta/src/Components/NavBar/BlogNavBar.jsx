import "./BlogNavBar.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import { Link, useParams } from "react-router-dom";
import { useTheme } from "../../context/ThemeContext";

const BlogNavBar = ({ isNotebook }) => {
  const { isDark, toggleTheme } = useTheme();
  const { filename } = useParams();
  const trimmedFilename =
    typeof filename === "string" ? filename.replace(/\.html$/, ".ipynb") : "";

  return (
    <nav className="blog-navbar">
      <ul className="blog-nav-menu">
        <Link className="blog-nav-link" to="/">
          <li>Home</li>
        </Link>
        <Link className="blog-nav-link" to="/blogs">
          <li>Blog</li>
        </Link>
      </ul>
      {isNotebook ? (
        <Link
          className="open-github-wrapper"
          to={
            "https://github.com/mjang01011/portfolio/blob/main/blog/models/" +
            trimmedFilename
          }
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className="open-github">Open on GitHub</span>
        </Link>
      ) : (
        <div className="blog-nav-right">
          <button
            className="theme-toggle"
            onClick={(e) => { e.stopPropagation(); toggleTheme(); }}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
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
      )}
    </nav>
  );
};

export default BlogNavBar;
