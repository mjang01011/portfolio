import "./Tools.css";
import brush_lightpurple from "../../assets/brush_lightpurple.png";

const Tools = () => {
  let skills = [
    "Python (PyTorch, Pandas, NumPy, Scikit-learn)",
    "Java",
    "Javascript (React.js, Express.js)",
    "SQL, MongoDB",
  ];
  return (
    <div className="tools">
      <div className="tools-title">
        <h1>Tools</h1>
        <img src={brush_lightpurple} alt="" />
      </div>
      <div>
        {skills.map((skill, index) => {
          return (
            <div key={index} className="tool">
              {skill}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Tools;
