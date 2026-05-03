import { Composition } from 'remotion';
import { EscapadeGourmandeVarC } from './compositions/EscapadeGourmandeVarC.js';
import { composition } from './brand/tokens.js';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="EscapadeGourmande_VarC"
        component={EscapadeGourmandeVarC}
        durationInFrames={composition.durationSeconds * composition.fps}
        fps={composition.fps}
        width={composition.width}
        height={composition.height}
      />
      {/* Future: VarA (ASMR Hero) and VarB (Reveal) registered here */}
    </>
  );
};
