import "./Hero.css";
import logo_github from "../../assets/logo_github.png";
import logo_linkedin from "../../assets/logo_linkedin.png";
import profile_img from "../../assets/headshot.jpg"
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <div className="hero" id="hero">
      <img src={profile_img} alt="headshot" />
      <h1>
        <span>Michael Jang</span>
      </h1>
      <p>
        I’m a 1st year Master’s student studying <span>Computer Science</span> at Stanford University, specializing in <span>AI</span>.
      </p>
      <p>
      </p>
      <p>
        I graduated from Duke University with a double major in <span>Electrical & Computer Engineering</span> and <span>Computer Science</span>, specializing in <span>Machine Learning</span> and <span>AI</span>. 
        Previously, I worked as a software engineering intern at <span>AMD</span>, developing an agentic chatbot to accelerate debugging of next-generation server processors at the Server Platform Debugging team using <span>MCP</span> and <span>RAG</span>.
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
          to="https://drive.google.com/file/d/1EwKzsuMgWOHzNtRril4SrrKjgsTDNDlO/view?usp=sharing"
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
