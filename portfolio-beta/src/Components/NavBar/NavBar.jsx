import "./NavBar.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import GoogleScholarIcon from "../Icons/GoogleScholarIcon";
import { Link } from "react-router-dom";
import { useTheme } from "../../context/ThemeContext";

const NavBar = () => {
  const { isDark, toggleTheme } = useTheme();
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (!element) return;
    const headerOffset = 80;
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.scrollY - headerOffset;

    window.scrollTo({
      top: offsetPosition,
      behavior: "smooth",
    });
  };

  return (
    <nav className="navbar">
      <ul className="nav-menu">
        <li onClick={() => scrollToSection("hero")}>Home</li>
        <li onClick={() => scrollToSection("about")}>About</li>
        <li onClick={() => scrollToSection("experiences")}>Experiences</li>
        <li onClick={() => scrollToSection("research")}>Research</li>
        <li onClick={() => scrollToSection("publications")}>Publications</li>
        <li onClick={() => scrollToSection("skills")}>Skills</li>
        <li onClick={() => scrollToSection("mywork")}>Projects</li>
        <li onClick={() => scrollToSection("blog")}>Blog</li>
      </ul>
      <div className="nav-right">
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
        <div className="nav-logo">
        <Link
          className="nav-link"
          to="https://www.linkedin.com/in/michaeljkjang/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="LinkedIn"
        >
          <img src={logo_linkedin} alt="LinkedIn" />
        </Link>
        <Link
          className="nav-link"
          to="https://github.com/mjang01011"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub"
        >
          <img src={logo_github} alt="GitHub" />
        </Link>
        <Link
          className="nav-link"
          to="https://scholar.google.com/citations?user=LizEmAIAAAAJ&hl=en"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Google Scholar"
        >
          <GoogleScholarIcon />
        </Link>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
