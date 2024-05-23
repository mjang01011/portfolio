import React from "react";
import "./Hero.css";
import profile_img from '../../assets/profile_img.svg'

const Hero = () => {
  return (
    <div className="hero">
      {/* <img src={profile_img} alt="" /> */}
      <h1><span>Michael Jang</span></h1>
      <p>I'm a senior at Duke University majoring in Electrical Computer Engineering and Computer Science.</p>
      <div className="hero-action">
        <div className="hero-connect">Connect With Me</div>
        <div className="hero-resume">My Resume</div>
      </div>
    </div>
  );
};

export default Hero;
