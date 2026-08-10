import "./Hero.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import profile_img from "../../assets/headshot.jpeg";
import stanford_logo from "../../assets/stanford_logo.png";
import duke_logo from "../../assets/duke_logo.png";
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <section className="hero" id="hero">
      <div className="hero-content">
        <div className="hero-image-wrap">
          <img src={profile_img} alt="Michael Jang" className="hero-image" />
        </div>
        <div className="hero-text">
          <h1>Michael Jang</h1>
          <p className="hero-tagline">
            Master's student studying <strong>Computer Science</strong> at Stanford, specializing in <strong>ML/AI</strong>.
          </p>
          <div className="hero-bio">
            <p>
              <img src={stanford_logo} alt="Stanford" className="hero-bio-logo" />
              MS Computer Science, Stanford &apos;27
            </p>
            <p>
              <img src={duke_logo} alt="Duke" className="hero-bio-logo" />
              BSE Electrical & Computer Engineering, Computer Science, Duke &apos;25
            </p>
          </div>
          <div className="hero-links">
            <Link
              className="hero-link"
              to="https://www.linkedin.com/in/michaeljkjang/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src={logo_linkedin} alt="LinkedIn" />
            </Link>
            <Link
              className="hero-link"
              to="https://github.com/mjang01011"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src={logo_github} alt="GitHub" />
            </Link>
          </div>
          <Link
            className="hero-resume"
            to="https://drive.google.com/file/d/1cRpAN6bhvlZQcMYLHNT41880Fo21kNKv/view?usp=sharing"
            target="_blank"
            rel="noopener noreferrer"
          >
            View Resume
          </Link>
        </div>
      </div>
    </section>
  );
};

export default Hero;
