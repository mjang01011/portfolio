import "./MyWork.css";
import theme_pattern from "../../assets/brush_purple.png";
import mywork_data from "../../assets/mywork_data";
import arrow_icon from "../../assets/arrow_icon.svg";
import { Link } from "react-router-dom";

const MyWork = () => {
  return (
    <div id="mywork" className="mywork">
      <div className="mywork-title">
        <h1>My latest work</h1>
        <img src={theme_pattern} alt="" />
      </div>
      <div className="mywork-container" id="mywork-container">
        {mywork_data.map((work, index) => {
          return (
            <div key={index} className="image-wrapper">
              <Link to={work.link} target="_blank" rel="noopener noreferrer">
                <img src={work.img} alt="" />
                <div className="overlay">
                  <div className="overlay-title">
                    {work.name}
                    <div className="overlay-stack">{work.stack}</div>
                  </div>
                </div>
              </Link>
            </div>
          );
        })}
      </div>
      <Link
        to="https://github.com/mjang01011"
        target="_blank"
        rel="noopener noreferrer"
      >
        <div className="mywork-showmore">
          <p>View my GitHub</p>
          <img src={arrow_icon} alt="" />
        </div>
      </Link>
    </div>
  );
};

export default MyWork;
