import "./Hero.css";
import profile_img from '../../assets/profile_img.svg'

const Hero = () => {
  return (
    <div className="hero">
      {/* <img src={profile_img} alt="" /> */}
      <h1><span>Michael Jang</span></h1>
      <p>I'm a senior at Duke University majoring in <span>Electrical Computer Engineering</span> and <span>Computer Science</span> with a concentration in <span>machine learning</span>.</p>
      <div className="hero-action">
        <div className="hero-connect">Connect With Me</div>
        <div className="hero-resume">My Resume</div>
      </div>
    </div>
  );
};

export default Hero;
