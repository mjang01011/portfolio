import "./Skills.css";

const Skills = () => {
  const skills = [
    { type: "Programming Languages", items: "Python, Java, Javascript" },
    { type: "Data Science & Machine Learning", items: "Python (PyTorch, Pandas, NumPy, Scikit-learn)" },
    { type: "Web Development", items: "Javascript (React.js, Express.js, Node.js)" },
    { type: "Databases", items: "SQL, MongoDB" },
    { type: "Tools", items: "Git, Wandb" },
  ];

  return (
    <section id="skills" className="skills">
      <div className="section-inner">
        <h2 className="section-title">Skills</h2>
        <div className="skills-grid">
          {skills.map((skill, index) => (
            <div className="skill-card" key={index}>
              <h3 className="skill-type">{skill.type}</h3>
              <p className="skill-items">{skill.items}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Skills;
