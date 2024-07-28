import "./Skills.css";
import brush_lightpurple from "../../assets/brush_lightpurple.png";

const Skills = () => {
  let skills = [
    "Python (PyTorch, Pandas, NumPy, Scikit-learn)",
    "Java",
    "Javascript (React.js, Express.js, Node.js)",
    "SQL, MongoDB",
  ];
  return (
    <div className="skills" id="skills">
      <div className="skills-title">
        <h1>Skills</h1>
        <img src={brush_lightpurple} alt="" />
      </div>
      <div className="skill-wrapper">
        {skills.map((skill, index) => {
          return (
            <div key={index} className="skill">
              {skill}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Skills;
