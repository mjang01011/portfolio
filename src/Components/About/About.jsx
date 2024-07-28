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
            <p>
              I am a passionate technologist with a keen interest in the wonders
              of both the digital and physical worlds. My journey spans the
              realms of machine learning, software development,
              astrophotography, and even the art of magic. Prior to my engineering studies,
              I was involved in astrophysics research internships.
            </p>
            <p>
              Apart from studies, you will often find me gazing at the night sky
              with my camera in hand. Astrophotography allows me to combine my
              love for technology and the cosmos, capturing the beauty of
              distant galaxies, star clusters, and nebulae. It’s a humbling
              experience that constantly reminds me of the vastness of the
              universe and our small, yet significant place within it. Magic has
              also been a source of wonder and joy in my life. Performing magic
              tricks is more than just a hobby; it’s a way to connect with
              people and bring a sense of awe and excitement into their lives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
