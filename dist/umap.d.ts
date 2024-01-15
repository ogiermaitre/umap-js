export declare type RandomFn = () => number;
export declare type EpochCallback = (epoch: number) => boolean | void;
export declare type Vector = number[];
export declare type Vectors = Vector[];
export interface UMAPParameters {
    learningRate?: number;
    localConnectivity?: number;
    minDist?: number;
    nComponents?: number;
    nEpochs?: number;
    nNeighbors?: number;
    negativeSampleRate?: number;
    repulsionStrength?: number;
    random?: RandomFn;
    setOpMixRatio?: number;
    spread?: number;
    transformQueueSize?: number;
}
export declare class UMAP {
    private learningRate;
    private localConnectivity;
    private minDist;
    private nComponents;
    private nEpochs;
    private nNeighbors;
    private negativeSampleRate;
    private random;
    private repulsionStrength;
    private setOpMixRatio;
    private spread;
    private c;
    private knnIndices?;
    private knnDistances?;
    private graph;
    private X;
    private isInitialized;
    private embedding;
    private optimizationState;
    constructor(params?: UMAPParameters);
    fit(X: Vectors): number[][];
    initializeFit(X: Vectors): number;
    private fastKnnIndices;
    private fuzzySimplicialSet;
    private smoothKNNDistance;
    private computeMembershipStrengths;
    private initializeSimplicialSetEmbedding;
    private makeEpochsPerSample;
    private assignOptimizationStateParameters;
    private prepareForOptimizationLoop;
    private initializeOptimization;
    private optimizeLayoutStep;
    private optimizeLayout;
    private getNEpochs;
}
export declare function findABParams(spread: number, minDist: number): {
    a: number;
    b: number;
};
