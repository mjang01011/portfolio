import "./NavBar.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import { Link } from "react-router-dom";

const NavBar = () => {
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    const headerOffset = 80;
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.scrollY - headerOffset;

    window.scrollTo({
      top: offsetPosition,
      behavior: "smooth",
    });
  };

  return (
    <div className="navbar">
      <ul className="nav-menu">
        <li onClick={() => scrollToSection("hero")}>Home</li>
        <li onClick={() => scrollToSection("about")}>About Me</li>
        <li onClick={() => scrollToSection("skills")}>Skills</li>
        <li onClick={() => scrollToSection("mywork")}>My Work</li>
        <li onClick={() => scrollToSection("blog")}>Blogs</li>
      </ul>
      {/* <div className="nav-connect">Connect With Me</div> */}
      <div className="nav-logo">
        <Link
          className="nav-link"
          to="https://www.linkedin.com/in/michaeljkjang/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src={logo_linkedin} alt="LinkedIn" />
        </Link>
        <Link
          className="nav-link"
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

export default NavBar;
