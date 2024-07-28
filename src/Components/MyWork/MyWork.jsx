import "./MyWork.css";
import theme_pattern from "../../assets/brush_purple.png";
import mywork_data from "../../assets/mywork_data";
import arrow_icon from "../../assets/arrow_icon.svg";

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
              <img src={work.img} alt="" />
              <div className="overlay">
                <div className="overlay-title">
                  {work.name}
                  <div className="overlay-stack">{work.stack}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mywork-showmore">
        <p>Show More</p>
        <img src={arrow_icon} alt="" />
      </div>
    </div>
  );
};

export default MyWork;
