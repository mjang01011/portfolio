import "./MyWork.css";
import mywork_data from "../../assets/mywork_data";
import { Link } from "react-router-dom";

const MyWork = () => {
  return (
    <section id="mywork" className="mywork">
      <div className="section-inner">
        <h2 className="section-title">Projects</h2>
        <div className="projects-grid">
          {mywork_data.map((work, index) => (
            <Link
              key={index}
              className="project-card"
              to={work.link}
              target="_blank"
              rel="noopener noreferrer"
            >
              <div className="project-image-wrap">
                <img src={work.img} alt={work.name} />
                <div className="project-overlay">
                  <span className="project-name">{work.name}</span>
                  <span className="project-stack">{work.stack}</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
        <Link
          className="github-link"
          to="https://github.com/mjang01011"
          target="_blank"
          rel="noopener noreferrer"
        >
          View on GitHub →
        </Link>
      </div>
    </section>
  );
};

export default MyWork;
