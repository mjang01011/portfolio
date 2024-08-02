import "./Hero.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <div className="hero" id="hero">
      {/* <img src={profile_img} alt="" /> */}
      <h1>
        <span>Michael Jang</span>
      </h1>
      <p>
        I&apos;m a senior at Duke University majoring in{" "}
        <span>Electrical Computer Engineering</span> and{" "}
        <span>Computer Science</span> with a concentration in{" "}
        <span>machine learning</span>.
      </p>
      <div className="logo">
        <Link
          className="link"
          to="https://www.linkedin.com/in/michaeljkjang/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src={logo_linkedin} alt="LinkedIn" />
        </Link>
        <Link
          className="link"
          to="https://github.com/mjang01011"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src={logo_github} alt="GitHub" />
        </Link>
      </div>
      <div className="hero-action">
        {/* <div className="hero-connect">Connect With Me</div> */}
        <Link
          className="link"
          to="https://drive.google.com/file/d/18DUDFr2Jk3W8arpDljsyB3YFM920DAHo/view?usp=sharing"
          target="_blank"
          rel="noopener noreferrer"
        >
          <div className="hero-resume">My Resume</div>
        </Link>
      </div>
    </div>
  );
};

export default Hero;
