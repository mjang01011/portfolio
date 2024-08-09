import "./Skills.css";
import brush_lightpurple from "../../assets/brush_lightpurple.png";

const Skills = () => {
  let skill_type = [
    "Programming Languages",
    "Data Science & Machine Learning",
    "Web Development",
    "Databases",
    "Tools",
  ];
  let skills = [
    "Python, Java, Javascript",
    "Python (PyTorch, Pandas, NumPy, Scikit-learn)",
    "Javascript (React.js, Express.js, Node.js)",
    "SQL, MongoDB",
    "Git, Wandb",
  ];
  return (
    <div className="skills" id="skills">
      <div className="skills-title">
        <h1>Skills</h1>
        <img src={brush_lightpurple} alt="" />
      </div>
      <div className="skill-wrapper">
        {skill_type.map((type, index) => {
          return (
            <a key={index} >
              <div className="type">
                {type}
              </div>
              <div className="skill">{skills[index]}</div>
            </a>
          );
        })}
      </div>
    </div>
  );
};

export default Skills;
