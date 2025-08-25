import "./About.css";
import brush_skyblue from "../../assets/brush_skyblue.png";

const About = () => {
  return (
    <div id="about" className="about">
      <div className="about-title">
        <h1>About Me</h1>
        <img src={brush_skyblue} alt="" />
      </div>
      <div className="about-sections">
        {/* <div className="about-left"> */}
        {/* <img src={profile_img} alt="" /> */}
        {/* </div> */}
        <div className="about-right">
          <div className="about-para">
          <p> I am a passionate technologist with a deep fascination for ML (LLMs in particular,
            with a focus on enhancing their inference efficiency). The challenge of optimizing 
            these models to balance performance and resource utilization excites me, as it merges cutting-edge 
            research with practical impact. This passion stems from my desire to make advanced AI systems more 
            accessible and scalable.
            I love to work in collaborative environments where continuous learning is encouraged, and I 
              actively seek opportunities to expand my skillset by integrating new tools into 
              my work. </p> <p> Outside of my technical pursuits, I am an avid traveler who loves exploring 
                to meet new people and capture the beauty of the world through 
                photography. Traveling provides me with fresh perspectives and the opportunity to connect 
                with people from all walks of life, enriching both my personal and professional growth. 
                I also enjoy astrophotography, which has been a hobby for me for the past 10 years. </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
