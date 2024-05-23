import React from "react";
import "./Blogs.css";
import theme_pattern from "../../assets/theme_pattern.svg";
import Services_Data from "../../assets/services_data";
import arrow_icon from "../../assets/arrow_icon.svg";

const Blogs = () => {
  return (
    <div className="blogs">
      <div className="blogs-title">
        <h1>My Blogs</h1>
        <img src={theme_pattern} alt="" />
      </div>
      <div className="blogs-container">
        {Services_Data.map((blog, index) => {
          return (
            <div key={index} className="blogs-format">
              <h3>{blog.s_no}</h3>
              <h2>{blog.s_name}</h2>
              <p>{blog.s_desc}</p>
              <div className="blogs-readmore">
                <p>Read More</p>
                <img src={arrow_icon} alt="" />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Blogs;
